#!/usr/bin/env python3
"""Quick test of the trained model to validate quality before deploying to ESP32"""

import json
import numpy as np
from keras import models

# Load model and vocabulary
model = models.load_model('model/dogberry_model_best.keras')

with open('model/vocab_correct.json', 'r') as f:
    vocab_data = json.load(f)
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}

vocab_size = len(idx_to_char)
print(f"Vocabulary size: {vocab_size}")
print(f"Model loaded: {model.count_params():,} parameters")

def generate_text(seed_text, length=150, temperature=0.8):
    """Generate text from seed"""
    generated = seed_text
    sequence = [char_to_idx.get(c, 0) for c in seed_text[-40:]]
    sequence = sequence + [0] * (40 - len(sequence))

    for _ in range(length):
        x = np.array([sequence])
        preds = model.predict(x, verbose=0)[0]

        # Apply temperature
        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        next_idx = np.random.choice(len(preds), p=preds)
        next_char = idx_to_char[next_idx]

        generated += next_char
        sequence = sequence[1:] + [next_idx]

    return generated

# Test with different seeds
print("\n" + "=" * 70)
print("TESTING MODEL GENERATION")
print("=" * 70)

seeds = [
    "Good morrow, friend @johnsonfarms.us! ",
    "Verily, I am Dogberry, the constable of ",
    "What ho! "
]

for seed in seeds:
    print(f"\nSeed: '{seed}'")
    result = generate_text(seed, length=100, temperature=0.8)
    print(f"Generated: {result}")
    print("-" * 70)
