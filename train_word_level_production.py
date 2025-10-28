#!/usr/bin/env python3
"""
Word-level Dogberry model - PRODUCTION VERSION
Train LSTM-256 model for 60 epochs (~4.5 hours) for final deployment
"""

import numpy as np
import json
import re
from keras import layers, models, callbacks, optimizers
import keras

print("=" * 70)
print("WORD-LEVEL DOGBERRY MODEL - PRODUCTION TRAINING")
print("=" * 70)

# Load corpus
print("\nLoading corpus...")
with open('corpus_word_level.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"  Corpus size: {len(text):,} characters")

# Tokenize into words
print("\nTokenizing...")
# Simple word tokenizer: split on whitespace and punctuation
words = re.findall(r"\w+|[.,!?;:]", text)

print(f"  Total words: {len(words):,}")

# Build vocabulary
word_counts = {}
for word in words:
    word = word.lower()  # Lowercase for consistency
    word_counts[word] = word_counts.get(word, 0) + 1

# Keep top N words, rest become <UNK>
# 4K vocabulary to fit model in ESP32's 8MB PSRAM
MAX_VOCAB = 4000
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
vocab_words = [w for w, c in sorted_words[:MAX_VOCAB-3]]  # -3 for special tokens

# Add special tokens
SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<START>']
vocab = SPECIAL_TOKENS + vocab_words

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"  Vocabulary size: {vocab_size:,} words")
print(f"  Most common: {', '.join(vocab_words[:10])}")

# Convert words to indices
word_indices = []
for word in words:
    word = word.lower()
    idx = word_to_idx.get(word, word_to_idx['<UNK>'])
    word_indices.append(idx)

print(f"  Converted to indices: {len(word_indices):,} words")

# Create training sequences
SEQ_LENGTH = 40  # Use 40 words of context
print(f"\nPreparing sequences (seq_length={SEQ_LENGTH})...")

sequences = []
next_words = []

for i in range(len(word_indices) - SEQ_LENGTH):
    sequences.append(word_indices[i:i+SEQ_LENGTH])
    next_words.append(word_indices[i+SEQ_LENGTH])

sequences = np.array(sequences)
next_words = np.array(next_words)

print(f"  Training sequences: {len(sequences):,}")

# Split train/val
val_split = 0.1
split_idx = int(len(sequences) * (1 - val_split))

X_train = sequences[:split_idx]
y_train = next_words[:split_idx]
X_val = sequences[split_idx:]
y_val = next_words[split_idx:]

print(f"  Train: {len(X_train):,} sequences")
print(f"  Val: {len(X_val):,} sequences")

# Build model - PRODUCTION (LSTM-256 trained longer)
print("\nBuilding model (Production - LSTM-256)...")

embedding_dim = 64
lstm_units = 256

model = models.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=SEQ_LENGTH),
    layers.LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.2),
    layers.Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Build model to get param count
model.build(input_shape=(None, SEQ_LENGTH))

total_params = model.count_params()
size_mb = (total_params * 4) / (1024 * 1024)

print(f"\nModel architecture:")
print(f"  Embedding({embedding_dim}) -> LSTM({lstm_units}) -> Dense({vocab_size})")
print(f"  Total parameters: {total_params:,}")
print(f"  Size (float32): {size_mb:.2f} MB")
print(f"  Fits in ESP32 8MB PSRAM: {'✓ YES' if size_mb < 8 else '✗ NO'}")

model.summary()

# Callbacks
checkpoint = callbacks.ModelCheckpoint(
    'model/dogberry_word_production_best.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00005,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train for 60 epochs
print("\n" + "=" * 70)
print("PRODUCTION TRAINING (60 epochs, ~4.5 hours)")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=128,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
)

# Save final model
model.save('model/dogberry_word_production.keras')
print("\n✓ Saved model/dogberry_word_production.keras")

# Save vocabulary
vocab_data = {
    'word_to_idx': word_to_idx,
    'idx_to_word': idx_to_word,
    'vocab_size': vocab_size,
    'seq_length': SEQ_LENGTH
}

with open('model/vocab_word_level_production.json', 'w') as f:
    json.dump(vocab_data, f, indent=2)

print("✓ Saved model/vocab_word_level_production.json")

# Save training history
history_data = {
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
}

with open('model/training_history_production.json', 'w') as f:
    json.dump(history_data, f, indent=2)

print("✓ Saved model/training_history_production.json")

# Test generation
print("\n" + "=" * 70)
print("TESTING GENERATION")
print("=" * 70)

def generate_text(seed_text, length=50, temperature=0.8):
    """Generate text from seed"""
    # Tokenize seed
    seed_words = re.findall(r"\w+|[.,!?;:]", seed_text.lower())
    sequence = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in seed_words[-SEQ_LENGTH:]]

    # Pad if needed
    while len(sequence) < SEQ_LENGTH:
        sequence.insert(0, word_to_idx['<PAD>'])

    result = seed_text

    for _ in range(length):
        x = np.array([sequence])
        preds = model.predict(x, verbose=0)[0]

        # Apply temperature
        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        next_idx = np.random.choice(len(preds), p=preds)
        next_word = idx_to_word[next_idx]

        if next_word not in ['<PAD>', '<UNK>', '<START>']:
            # Add space before word unless it's punctuation
            if next_word in [',', '.', '!', '?', ';', ':']:
                result += next_word
            else:
                result += ' ' + next_word

        # Update sequence
        sequence = sequence[1:] + [next_idx]

    return result

# Test with different seeds
print("\nGenerating test samples with temperature=0.8:")
print("-" * 70)

seeds = [
    "Good morrow, friend!",
    "I am Dogberry, the constable of",
    "What ho!",
    "Truly, I say unto thee,",
    "The watch must be vigilant"
]

for seed in seeds:
    print(f"\nSeed: '{seed}'")
    generated = generate_text(seed, length=40, temperature=0.8)
    print(f"Generated: {generated}")
    print("-" * 70)

# Also test with lower temperature (more conservative)
print("\n\nGenerating test samples with temperature=0.5 (more conservative):")
print("-" * 70)

for seed in seeds[:3]:  # Just first 3
    print(f"\nSeed: '{seed}'")
    generated = generate_text(seed, length=40, temperature=0.5)
    print(f"Generated: {generated}")
    print("-" * 70)

print("\n" + "=" * 70)
print("✓ PRODUCTION TRAINING COMPLETE!")
print("=" * 70)
print("\nModel ready for ESP32 deployment:")
print(f"  - Model file: model/dogberry_word_production_best.keras")
print(f"  - Vocabulary: model/vocab_word_level_production.json")
print(f"  - Model size: {size_mb:.2f} MB (fits in 8MB PSRAM)")
print(f"  - Total parameters: {total_params:,}")
print("\nNext steps:")
print("  1. Export model to ESP32 C header format")
print("  2. Update ESP32 firmware for word-level inference")
print("  3. Update bot reply interval to 60 seconds")
print("  4. Deploy to hardware and test")
