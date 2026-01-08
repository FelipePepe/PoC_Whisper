"""
Interfaz gráfica de escritorio para transcripción de audio con Whisper
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
from pathlib import Path
from .transcribe import transcribe_audio, save_transcription
from .diarize import transcribe_with_speaker_diarization, format_transcription_by_speaker, save_diarized_transcription
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper - Transcripción de Audio")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.audio_file = None
        self.processing = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="🎙️ Transcripción de Audio con Whisper", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Selección de archivo
        file_frame = ttk.LabelFrame(main_frame, text="Archivo de Audio", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(0, weight=1)
        
        self.file_label = ttk.Label(file_frame, text="Ningún archivo seleccionado", 
                                    foreground="gray")
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        ttk.Button(file_frame, text="Seleccionar Archivo", 
                  command=self.select_file).grid(row=0, column=1, padx=5)
        
        # Opciones de transcripción
        options_frame = ttk.LabelFrame(main_frame, text="Opciones", padding="10")
        options_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Tipo de transcripción
        ttk.Label(options_frame, text="Tipo:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.transcription_type = tk.StringVar(value="simple")
        ttk.Radiobutton(options_frame, text="Simple (más rápido)", 
                       variable=self.transcription_type, 
                       value="simple").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="Con identificación de hablantes (lento)", 
                       variable=self.transcription_type, 
                       value="diarization").grid(row=0, column=2, sticky=tk.W)
        
        # Modelo
        ttk.Label(options_frame, text="Modelo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(options_frame, textvariable=self.model_var, 
                                   values=["tiny", "base", "small", "medium", "large"],
                                   state="readonly", width=15)
        model_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Idioma
        ttk.Label(options_frame, text="Idioma:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.language_var = tk.StringVar(value="es")
        lang_combo = ttk.Combobox(options_frame, textvariable=self.language_var,
                                 values=["auto", "es", "en", "fr", "de", "it", "pt"],
                                 state="readonly", width=10)
        lang_combo.grid(row=1, column=3, sticky=tk.W)
        
        # Botones de acción
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.transcribe_btn = ttk.Button(button_frame, text="▶️ Transcribir", 
                                         command=self.start_transcription,
                                         style='Accent.TButton')
        self.transcribe_btn.pack(side=tk.LEFT, padx=5)
        
        self.cancel_btn = ttk.Button(button_frame, text="⏹️ Cancelar", 
                                     command=self.cancel_transcription,
                                     state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="💾 Guardar Resultado", 
                  command=self.save_result).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="🗑️ Limpiar", 
                  command=self.clear_result).pack(side=tk.LEFT, padx=5)
        
        # Barra de progreso
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Área de resultados
        result_frame = ttk.LabelFrame(main_frame, text="Resultado de la Transcripción", 
                                     padding="10")
        result_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, 
                                                     height=20, font=('Arial', 10))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Barra de estado
        self.status_bar = ttk.Label(main_frame, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    def select_file(self):
        """Seleccionar archivo de audio"""
        filetypes = (
            ('Archivos de audio', '*.mp3 *.wav *.m4a *.mp4 *.webm *.ogg *.flac'),
            ('Todos los archivos', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Seleccionar archivo de audio',
            filetypes=filetypes
        )
        
        if filename:
            self.audio_file = filename
            self.file_label.config(text=Path(filename).name, foreground="black")
            self.status_bar.config(text=f"Archivo seleccionado: {Path(filename).name}")
    
    def start_transcription(self):
        """Iniciar el proceso de transcripción"""
        if not self.audio_file:
            messagebox.showwarning("Archivo no seleccionado", 
                                 "Por favor, selecciona un archivo de audio primero.")
            return
        
        if not os.path.exists(self.audio_file):
            messagebox.showerror("Error", "El archivo seleccionado no existe.")
            return
        
        # Verificar token si es diarización
        if self.transcription_type.get() == "diarization":
            if not os.getenv('HF_TOKEN'):
                messagebox.showerror("Token no encontrado", 
                                   "Para usar diarización necesitas configurar HF_TOKEN en el archivo .env")
                return
        
        self.processing = True
        self.transcribe_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.progress_bar.start(10)
        
        # Ejecutar en thread separado
        thread = threading.Thread(target=self.process_audio, daemon=True)
        thread.start()
    
    def process_audio(self):
        """Procesar el audio en un thread separado"""
        try:
            model = self.model_var.get()
            language = self.language_var.get() if self.language_var.get() != "auto" else None
            
            if self.transcription_type.get() == "simple":
                self.update_status("Transcribiendo audio...")
                result = transcribe_audio(self.audio_file, model, language)
                text = result["text"]
                
                self.root.after(0, self.show_result, text)
                self.root.after(0, self.update_status, "Transcripción completada ✓")
                
            else:  # diarization
                self.update_status("Identificando hablantes (esto puede tardar)...")
                hf_token = os.getenv('HF_TOKEN')
                segments = transcribe_with_speaker_diarization(
                    self.audio_file, hf_token, model, language
                )
                text = format_transcription_by_speaker(segments)
                
                self.root.after(0, self.show_result, text)
                self.root.after(0, self.update_status, "Transcripción con diarización completada ✓")
        
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
        
        finally:
            self.root.after(0, self.finish_processing)
    
    def show_result(self, text):
        """Mostrar resultado en el área de texto"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
    
    def show_error(self, error_msg):
        """Mostrar error"""
        messagebox.showerror("Error", f"Error durante la transcripción:\n{error_msg}")
        self.update_status(f"Error: {error_msg}")
    
    def update_status(self, message):
        """Actualizar mensaje de estado"""
        self.progress_label.config(text=message)
        self.status_bar.config(text=message)
    
    def finish_processing(self):
        """Finalizar el procesamiento"""
        self.processing = False
        self.transcribe_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.progress_bar.stop()
    
    def cancel_transcription(self):
        """Cancelar la transcripción (limitado en threading)"""
        if messagebox.askyesno("Cancelar", "¿Estás seguro de que quieres cancelar?"):
            self.update_status("Cancelando...")
            # Nota: Es difícil cancelar threads en Python, esto solo actualiza la UI
            self.finish_processing()
    
    def save_result(self):
        """Guardar el resultado en un archivo"""
        text = self.result_text.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Sin resultado", "No hay texto para guardar.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Guardado", f"Archivo guardado en:\n{filename}")
                self.status_bar.config(text=f"Guardado en: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar:\n{e}")
    
    def clear_result(self):
        """Limpiar el área de resultados"""
        self.result_text.delete(1.0, tk.END)
        self.status_bar.config(text="Resultado limpiado")


def main():
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
