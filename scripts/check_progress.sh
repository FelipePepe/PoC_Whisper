#!/bin/bash
# Script para verificar el progreso de la diarización

PID=19940
LOG_FILE="diarize_output.log"

echo "========================================"
echo "Monitor de Progreso - Diarización"
echo "========================================"
echo ""

# Verificar si el proceso está corriendo
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Proceso ACTIVO (PID: $PID)"
    echo ""
    
    # Mostrar uso de recursos
    echo "Uso de recursos:"
    ps -p $PID -o %cpu,%mem,etime,cmd --no-headers
    echo ""
    
    # Mostrar últimas líneas del log
    echo "Últimas líneas del log:"
    echo "----------------------------------------"
    if [ -f "$LOG_FILE" ]; then
        tail -20 "$LOG_FILE"
    else
        echo "Log no disponible aún..."
    fi
    echo "----------------------------------------"
    echo ""
    echo "Para ver el log en tiempo real: tail -f $LOG_FILE"
    echo "Para detener el proceso: kill $PID"
else
    echo "❌ Proceso TERMINADO o NO ENCONTRADO"
    echo ""
    
    # Mostrar el log completo si existe
    if [ -f "$LOG_FILE" ]; then
        echo "Contenido del log:"
        echo "----------------------------------------"
        cat "$LOG_FILE"
        echo "----------------------------------------"
    fi
    
    # Verificar si se generaron archivos de salida
    echo ""
    echo "Archivos generados:"
    ls -lh *_diarized_*.txt 2>/dev/null || echo "No se encontraron archivos de transcripción"
fi

echo ""
echo "========================================"
