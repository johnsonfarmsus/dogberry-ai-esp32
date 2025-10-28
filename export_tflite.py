"""
Export the trained Dogberry model to TensorFlow Lite format
"""

import tensorflow as tf
from tensorflow import keras
import json

print("Loading trained model...")
model = keras.models.load_model('model/dogberry_model.keras')

print("\nModel summary:")
model.summary()

print("\nConverting to TensorFlow Lite (with TF ops for LSTM)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# LSTM requires TensorFlow ops in TFLite
# This is supported by TensorFlow Lite Micro on ESP32
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Use TFLite builtins
    tf.lite.OpsSet.SELECT_TF_OPS     # Include TF ops for LSTM
]
converter._experimental_lower_tensor_list_ops = False

# Use dynamic range quantization for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save TFLite model
tflite_path = 'model/dogberry_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size_mb = len(tflite_model) / (1024 * 1024)
tflite_size_kb = len(tflite_model) / 1024

print(f"\n✓ TFLite model saved: {tflite_path}")
print(f"  Size: {tflite_size_kb:.1f} KB ({len(tflite_model):,} bytes)")
print(f"  Size: {tflite_size_mb:.2f} MB")

# Verify vocab.json exists
try:
    with open('model/vocab.json', 'r') as f:
        vocab = json.load(f)
    print(f"\n✓ Vocabulary: {vocab['vocab_size']} characters")
    print(f"✓ Sequence length: {vocab['seq_length']}")
except FileNotFoundError:
    print("\n⚠️  Warning: vocab.json not found")

print("\n" + "=" * 70)
print("✓ MODEL READY FOR ESP32!")
print("\nNext steps:")
print("1. Copy model/dogberry_model.tflite to ../dogberry-bot/esp32_inference/data/")
print("2. Copy model/vocab.json to ../dogberry-bot/esp32_inference/data/")
print("3. Upload to ESP32 via SPIFFS")
print("=" * 70)
