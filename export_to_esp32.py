#!/usr/bin/env python3
"""
Export word-level Dogberry model to ESP32 C header format
"""

import numpy as np
import json
from keras import models

print("=" * 70)
print("EXPORT WORD-LEVEL MODEL TO ESP32")
print("=" * 70)

# Load the trained model
print("\nLoading model...")
model = models.load_model('model/dogberry_word_production_best.keras')

print("✓ Model loaded")

# Load vocabulary
print("\nLoading vocabulary...")
with open('model/vocab_word_level_production.json', 'r') as f:
    vocab_data = json.load(f)

word_to_idx = vocab_data['word_to_idx']
idx_to_word = vocab_data['idx_to_word']
vocab_size = vocab_data['vocab_size']
seq_length = vocab_data['seq_length']

print(f"✓ Vocabulary: {vocab_size} words")
print(f"✓ Sequence length: {seq_length} words")

# Get model layers
embedding_layer = model.layers[0]
lstm_layer = model.layers[1]
dense_layer = model.layers[2]

# Extract weights
print("\nExtracting model weights...")
embedding_weights = embedding_layer.get_weights()[0]  # shape: (vocab_size, embedding_dim)
lstm_weights = lstm_layer.get_weights()  # [kernel, recurrent_kernel, bias]
dense_weights = dense_layer.get_weights()  # [kernel, bias]

embedding_dim = embedding_weights.shape[1]
lstm_units = lstm_weights[1].shape[1] // 4  # Recurrent kernel has 4x units (i,f,c,o gates)

print(f"  Embedding: {vocab_size} x {embedding_dim}")
print(f"  LSTM: {lstm_units} units")
print(f"  Dense: {lstm_units} x {vocab_size}")

# Calculate total parameters
total_params = model.count_params()
size_mb = (total_params * 4) / (1024 * 1024)
print(f"\n  Total parameters: {total_params:,}")
print(f"  Size (float32): {size_mb:.2f} MB")

# Export vocabulary to C header
print("\nExporting vocabulary...")
vocab_header = f"""#ifndef VOCAB_DATA_H
#define VOCAB_DATA_H

// Vocabulary configuration
#define VOCAB_SIZE {vocab_size}
#define SEQ_LENGTH {seq_length}

// Word-to-index mapping
const char* VOCAB_WORDS[{vocab_size}] = {{
"""

# Sort by index to maintain order
for i in range(vocab_size):
    word = idx_to_word[str(i)]
    # Escape special characters
    word_escaped = word.replace('\\', '\\\\').replace('"', '\\"')
    vocab_header += f'    "{word_escaped}"'
    if i < vocab_size - 1:
        vocab_header += ','
    vocab_header += f'  // {i}\n'

vocab_header += "};\n\n#endif // VOCAB_DATA_H\n"

# Write vocab header
vocab_path = '/Users/trevorjohnson/Documents/Projects/dogberry-bot/esp32_firmware/src/vocab_data_word.h'
with open(vocab_path, 'w') as f:
    f.write(vocab_header)

print(f"✓ Saved {vocab_path}")

# Export model weights to C header
print("\nExporting model weights...")

def array_to_c(arr, name, indent=1):
    """Convert numpy array to C array string with PROGMEM attribute"""
    flat = arr.flatten()
    tab = "    " * indent
    c_str = f"const float {name}[{len(flat)}] PROGMEM = {{\n"

    # Write in rows of 8 values for readability
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        c_str += tab + ", ".join([f"{x:.8f}f" for x in chunk])
        if i + 8 < len(flat):
            c_str += ","
        c_str += "\n"

    c_str += "};\n"
    return c_str

weights_header = f"""#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

// Model architecture
#define EMBEDDING_DIM {embedding_dim}
#define LSTM_UNITS {lstm_units}

// Embedding weights: {embedding_weights.shape}
"""

weights_header += array_to_c(embedding_weights, "EMBEDDING_WEIGHTS")
weights_header += "\n"

# LSTM weights
# LSTM has 3 weight matrices: kernel, recurrent_kernel, bias
# Each has 4 sets of weights for gates: input, forget, cell, output
lstm_kernel = lstm_weights[0]  # shape: (embedding_dim, lstm_units * 4)
lstm_recurrent = lstm_weights[1]  # shape: (lstm_units, lstm_units * 4)
lstm_bias = lstm_weights[2]  # shape: (lstm_units * 4,)

weights_header += f"// LSTM kernel: {lstm_kernel.shape}\n"
weights_header += array_to_c(lstm_kernel, "LSTM_KERNEL")
weights_header += "\n"

weights_header += f"// LSTM recurrent: {lstm_recurrent.shape}\n"
weights_header += array_to_c(lstm_recurrent, "LSTM_RECURRENT")
weights_header += "\n"

weights_header += f"// LSTM bias: {lstm_bias.shape}\n"
weights_header += array_to_c(lstm_bias, "LSTM_BIAS")
weights_header += "\n"

# Dense weights
dense_kernel = dense_weights[0]  # shape: (lstm_units, vocab_size)
dense_bias = dense_weights[1]  # shape: (vocab_size,)

weights_header += f"// Dense kernel: {dense_kernel.shape}\n"
weights_header += array_to_c(dense_kernel, "DENSE_KERNEL")
weights_header += "\n"

weights_header += f"// Dense bias: {dense_bias.shape}\n"
weights_header += array_to_c(dense_bias, "DENSE_BIAS")
weights_header += "\n"

weights_header += "#endif // MODEL_WEIGHTS_H\n"

# Write weights header
weights_path = '/Users/trevorjohnson/Documents/Projects/dogberry-bot/esp32_firmware/src/model_weights_word.h'
with open(weights_path, 'w') as f:
    f.write(weights_header)

print(f"✓ Saved {weights_path}")

# Create a summary file
summary = f"""WORD-LEVEL MODEL EXPORT SUMMARY
{'=' * 70}

Model Architecture:
  - Embedding({embedding_dim})
  - LSTM({lstm_units})
  - Dense({vocab_size})

Parameters:
  - Total: {total_params:,}
  - Size: {size_mb:.2f} MB

Vocabulary:
  - Size: {vocab_size} words
  - Sequence length: {seq_length} words
  - Special tokens: <PAD>, <UNK>, <START>

Output Files:
  - Vocabulary: {vocab_path}
  - Weights: {weights_path}

Next Steps:
  1. Update DogberryAI.h with new dimensions:
     #define VOCAB_SIZE {vocab_size}
     #define SEQ_LENGTH {seq_length}
     #define EMBEDDING_DIM {embedding_dim}
     #define LSTM_UNITS {lstm_units}

  2. Update DogberryAI.cpp to use word-level tokenization

  3. Update main.cpp mention check interval to 60 seconds

  4. Build and flash firmware
"""

with open('model/export_summary.txt', 'w') as f:
    f.write(summary)

print("\n" + "=" * 70)
print("✓ EXPORT COMPLETE!")
print("=" * 70)
print(summary)
