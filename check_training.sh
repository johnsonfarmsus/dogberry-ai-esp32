#!/bin/bash

# Training Status Checker
# Quick script to check Shakespeare training progress

echo "========================================"
echo "DOGBERRY MODEL TRAINING STATUS"
echo "========================================"
echo ""

# Check if training is running
TRAIN_PID=$(ps aux | grep "train_shakespeare.py" | grep -v grep | awk '{print $2}')

if [ -z "$TRAIN_PID" ]; then
    echo "❌ Training not running"
    echo ""

    # Check if model exists (training completed)
    if [ -f "model/shakespeare_model.keras" ]; then
        echo "✅ Base model found: model/shakespeare_model.keras"
        ls -lh model/shakespeare_model.keras
        echo ""
        echo "Ready for fine-tuning!"
        echo "Next: python3 finetune_dogberry.py"
    else
        echo "No base model found."
        echo "Start training: python3 train_shakespeare.py"
    fi
else
    echo "✅ Training is running (PID: $TRAIN_PID)"
    echo ""

    # Show CPU usage
    echo "CPU Usage:"
    ps -p $TRAIN_PID -o %cpu,etime,command | tail -n 1
    echo ""

    # Show memory usage
    echo "Memory Usage:"
    ps -p $TRAIN_PID -o %mem,rss,command | tail -n 1
    echo ""

    # Estimate progress if we can find epoch info
    echo "Progress:"
    echo "(Training runs for 40 epochs, ~6 hours total)"
    echo ""

    # Show recent activity (if available)
    if [ -f "shakespeare_training.log" ]; then
        echo "Recent log entries:"
        tail -n 5 shakespeare_training.log
    else
        echo "No training log found (output to stdout)"
    fi
fi

echo ""
echo "========================================"
echo "Files Status:"
echo "========================================"
echo ""

# Check what files exist
[ -f "model/shakespeare_model.keras" ] && echo "✅ shakespeare_model.keras (base model)" || echo "⏳ shakespeare_model.keras (pending)"
[ -f "model/shakespeare_model.tflite" ] && echo "✅ shakespeare_model.tflite (ESP32 ready)" || echo "⏳ shakespeare_model.tflite (pending)"
[ -f "model/vocab.json" ] && echo "✅ vocab.json" || echo "⏳ vocab.json (pending)"
[ -f "dogberry_corpus_combined.txt" ] && echo "✅ dogberry_corpus_combined.txt (fine-tuning ready)" || echo "⏳ dogberry_corpus_combined.txt (need to prepare)"
[ -f "shakespeare_dogberry_finetuned.keras" ] && echo "✅ shakespeare_dogberry_finetuned.keras (final model)" || echo "⏳ shakespeare_dogberry_finetuned.keras (pending)"

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""

if [ ! -f "model/shakespeare_model.keras" ]; then
    echo "1. Wait for base training to complete (~6 hours)"
    echo "2. Generate synthetic Dogberry corpus (see FINETUNING_WORKFLOW.md)"
    echo "3. Run: python3 prepare_finetuning_data.py"
    echo "4. Run: python3 finetune_dogberry.py"
elif [ ! -f "dogberry_corpus_combined.txt" ]; then
    echo "1. Generate synthetic Dogberry corpus (see DOGBERRY_GENERATION_GUIDE.md)"
    echo "2. Run: python3 prepare_finetuning_data.py"
    echo "3. Run: python3 finetune_dogberry.py"
elif [ ! -f "shakespeare_dogberry_finetuned.keras" ]; then
    echo "1. Run: python3 finetune_dogberry.py"
    echo "2. Run: python3 export_tflite.py"
    echo "3. Deploy to ESP32"
else
    echo "✅ Training complete!"
    echo "Deploy to ESP32:"
    echo "  cp model/*.tflite ../dogberry-bot/esp32_inference/data/"
    echo "  cp model/vocab.json ../dogberry-bot/esp32_inference/data/"
fi

echo ""
