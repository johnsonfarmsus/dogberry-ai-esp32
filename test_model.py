#!/usr/bin/env python3
"""
Test the trained model to see if it generates coherent text
"""

from tensorflow import keras
import numpy as np
import json

print("=" * 70)
print("TESTING DOGBERRY MODEL")
print("=" * 70)

# Load model
print("\nLoading model...")
model = keras.models.load_model('model/dogberry_model.keras')

# Load vocab
with open('model/vocab_correct.json') as f:
    vocab = json.load(f)
    char_to_idx = vocab['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
    vocab_size = vocab['vocab_size']
    seq_length = vocab['seq_length']

print(f"Vocabulary size: {vocab_size}")
print(f"Sequence length: {seq_length}")

def generate_text(seed, length=200, temperature=0.8):
    """Generate text from seed"""
    # Pad seed
    if len(seed) < seq_length:
        seed = ' ' * (seq_length - len(seed)) + seed

    seed = seed[-seq_length:]
    generated = seed

    for _ in range(length):
        # Encode current sequence
        x = np.array([[char_to_idx.get(ch, 0) for ch in generated[-seq_length:]]])

        # Predict
        predictions = model.predict(x, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # Sample
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

        # Stop at sentence end
        if next_char in '.!?' and len(generated) > len(seed) + 30:
            break

    return generated

# Test with different seeds
seeds = [
    "Good morrow, friend",
    "I am Dogberry",
    "You are to comprehend",
    "Masters, remember"
]

print("\n" + "=" * 70)
print("GENERATION TEST")
print("=" * 70)

for seed in seeds:
    print(f"\nSeed: \"{seed}\"")
    print("-" * 70)
    result = generate_text(seed, length=150, temperature=0.8)
    print(result)
    print()

print("\n" + "=" * 70)
print("If the above looks good, the model works in Python.")
print("If it's gibberish, we need to retrain the model.")
print("=" * 70)
