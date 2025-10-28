#!/usr/bin/env python3
"""
Retrain Dogberry model with the full combined corpus
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import json

print("=" * 70)
print("RETRAINING DOGBERRY MODEL WITH FULL CORPUS")
print("=" * 70)

# Load combined corpus
print("\nLoading combined corpus...")
with open('dogberry_corpus_combined.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Corpus size: {len(text):,} characters")

# Build vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {vocab_size} unique characters")

# Parameters
seq_length = 40
embedding_dim = 48  # Increased from 32
lstm_units = 192    # Increased from 128
batch_size = 256
epochs = 60         # More epochs

print(f"\nModel architecture:")
print(f"  Sequence length: {seq_length}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  LSTM units: {lstm_units}")

# Prepare training data
print("\nPreparing training sequences...")
encoded = np.array([char_to_idx[ch] for ch in text])

sequences = []
next_chars = []

step = 3
for i in range(0, len(encoded) - seq_length, step):
    sequences.append(encoded[i:i + seq_length])
    next_chars.append(encoded[i + seq_length])

X = np.array(sequences)
y = np.array(next_chars)

# Split train/val
split_idx = int(len(X) * 0.9)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Training sequences: {len(X_train):,}")
print(f"Validation sequences: {len(X_val):,}")

# Build model
print("\nBuilding model...")
model = keras.Sequential([
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=seq_length
    ),
    layers.LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.2),
    layers.Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Build model before counting params
model.build(input_shape=(None, seq_length))

total_params = model.count_params()
size_mb = (total_params * 4) / (1024 * 1024)

print(f"\nModel statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Size (float32): {size_mb:.2f} MB")

model.summary()

# Train
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'model/dogberry_model_best.keras',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=0.0001
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model.save('model/dogberry_model.keras')
print("✓ Saved model/dogberry_model.keras")

# Save vocabulary
vocab_data = {
    'char_to_idx': char_to_idx,
    'idx_to_char': {int(k): v for k, v in idx_to_char.items()},
    'vocab_size': vocab_size,
    'seq_length': seq_length
}

with open('model/vocab_correct.json', 'w') as f:
    json.dump(vocab_data, f, indent=2)

print("✓ Saved model/vocab_correct.json")

# Test generation
print("\n" + "=" * 70)
print("TESTING GENERATION")
print("=" * 70)

def generate_text(seed, length=200, temperature=0.8):
    if len(seed) < seq_length:
        seed = ' ' * (seq_length - len(seed)) + seed
    seed = seed[-seq_length:]
    generated = seed

    for _ in range(length):
        x = np.array([[char_to_idx.get(ch, 0) for ch in generated[-seq_length:]]])
        predictions = model.predict(x, verbose=0)[0]

        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

        if next_char in '.!?' and len(generated) > len(seed) + 30:
            break

    return generated

seeds = ["Good morrow, friend", "I am Dogberry", "Masters, remember"]

for seed in seeds:
    print(f"\nSeed: \"{seed}\"")
    print("-" * 70)
    result = generate_text(seed, length=150, temperature=0.8)
    print(result)

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("Next: Run export_weights.py to update ESP32")
print("=" * 70)
