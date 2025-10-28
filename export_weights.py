#!/usr/bin/env python3
"""
Export trained Dogberry model weights to simple format for ESP32
"""

import numpy as np
import json
from tensorflow import keras

print("=" * 70)
print("EXPORTING DOGBERRY MODEL WEIGHTS FOR ESP32")
print("=" * 70)

# Load model
print("\nLoading model...")
model = keras.models.load_model('model/dogberry_model_best.keras')

print("✓ Model loaded")
print(f"  Layers: {len(model.layers)}")

# Get layer names
for i, layer in enumerate(model.layers):
    print(f"  {i}: {layer.name} - {layer.__class__.__name__}")

# Extract weights
print("\nExtracting weights...")

# Layer 0: Embedding
embedding_weights = model.layers[0].get_weights()[0]  # Shape: (vocab_size, embedding_dim)
print(f"  Embedding: {embedding_weights.shape}")

# Layer 1: LSTM
lstm_weights = model.layers[1].get_weights()
# LSTM has 3 weight matrices: input, recurrent, and bias
# weights[0]: input kernel (embedding_dim, 4*units) - for input->hidden
# weights[1]: recurrent kernel (units, 4*units) - for hidden->hidden
# weights[2]: bias (4*units,)

kernel = lstm_weights[0]  # Input kernel
recurrent = lstm_weights[1]  # Recurrent kernel
bias = lstm_weights[2]  # Bias

print(f"  LSTM kernel: {kernel.shape}")
print(f"  LSTM recurrent: {recurrent.shape}")
print(f"  LSTM bias: {bias.shape}")

# Layer 2: Dense
dense_weights = model.layers[2].get_weights()
dense_kernel = dense_weights[0]  # Shape: (lstm_units, vocab_size)
dense_bias = dense_weights[1]  # Shape: (vocab_size,)

print(f"  Dense kernel: {dense_kernel.shape}")
print(f"  Dense bias: {dense_bias.shape}")

# Load vocab
with open('model/vocab_correct.json') as f:
    vocab = json.load(f)

# Save as NPZ (compressed numpy format)
print("\nSaving weights...")
np.savez_compressed(
    'model/weights.npz',
    embedding=embedding_weights,
    lstm_kernel=kernel,
    lstm_recurrent=recurrent,
    lstm_bias=bias,
    dense_kernel=dense_kernel,
    dense_bias=dense_bias
)

print("✓ Saved to model/weights.npz")

# Calculate total size
total_params = (embedding_weights.size + kernel.size + recurrent.size +
                bias.size + dense_kernel.size + dense_bias.size)
size_mb = (total_params * 4) / (1024 * 1024)  # float32

vocab_size = len(vocab['char_to_idx'])
seq_length = 40  # Fixed sequence length

print(f"\nModel statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Size (float32): {size_mb:.2f} MB")
print(f"  Vocab size: {vocab_size}")
print(f"  Sequence length: {seq_length}")

# Now create a simple C-friendly format
print("\nCreating C++ header file...")

def array_to_c(arr, name, dtype="float"):
    """Convert numpy array to C array declaration"""
    flat = arr.flatten()

    lines = []
    lines.append(f"const {dtype} {name}[{len(flat)}] = {{")

    # Write in rows of 8 values
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        values = ", ".join(f"{v:.8f}f" for v in chunk)
        lines.append(f"    {values},")

    lines.append("};")
    return "\n".join(lines)

# Write header file
with open('/Users/trevorjohnson/Documents/Projects/dogberry-bot/esp32_firmware/src/model_weights.h', 'w') as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n")
    f.write("#define MODEL_WEIGHTS_H\n\n")
    f.write("// Auto-generated model weights for Dogberry AI\n")
    f.write(f"// Total parameters: {total_params:,}\n")
    f.write(f"// Model size: {size_mb:.2f} MB\n\n")

    # Dimensions
    embedding_dim = embedding_weights.shape[1]
    lstm_units = recurrent.shape[0]

    f.write(f"#define VOCAB_SIZE {vocab_size}\n")
    f.write(f"#define SEQ_LENGTH {seq_length}\n")
    f.write(f"#define EMBEDDING_DIM {embedding_dim}\n")
    f.write(f"#define LSTM_UNITS {lstm_units}\n\n")

    # Write arrays
    f.write("// Embedding weights\n")
    f.write(array_to_c(embedding_weights, "embedding_weights"))
    f.write("\n\n")

    f.write("// LSTM kernel (input->hidden)\n")
    f.write(array_to_c(kernel, "lstm_kernel"))
    f.write("\n\n")

    f.write("// LSTM recurrent (hidden->hidden)\n")
    f.write(array_to_c(recurrent, "lstm_recurrent"))
    f.write("\n\n")

    f.write("// LSTM bias\n")
    f.write(array_to_c(bias, "lstm_bias"))
    f.write("\n\n")

    f.write("// Dense kernel (LSTM->output)\n")
    f.write(array_to_c(dense_kernel, "dense_kernel"))
    f.write("\n\n")

    f.write("// Dense bias\n")
    f.write(array_to_c(dense_bias, "dense_bias"))
    f.write("\n\n")

    f.write("#endif // MODEL_WEIGHTS_H\n")

print("✓ Created model_weights.h")

print("\n" + "=" * 70)
print("✓ EXPORT COMPLETE!")
print("\nFiles created:")
print("  - model/weights.npz (for testing)")
print("  - src/model_weights.h (for ESP32)")
print("=" * 70)
