"""
Test the trained Dogberry model - see what he has to say!
"""

from train_dogberry import DogberryModel
from tensorflow import keras
import json

print("=" * 70)
print("DOGBERRY TEXT GENERATION TEST")
print("=" * 70)

# Load model
print("\nLoading model...")
model_obj = DogberryModel()
model_obj.model = keras.models.load_model('model/dogberry_model.keras')

# Load vocabulary
with open('model/vocab.json') as f:
    vocab = json.load(f)
    model_obj.char_to_idx = vocab['char_to_idx']
    model_obj.idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
    model_obj.vocab_size = vocab['vocab_size']
    model_obj.seq_length = vocab['seq_length']

print(f"✓ Model loaded ({model_obj.vocab_size} chars, seq_length={model_obj.seq_length})")

# Test seeds
test_seeds = [
    "Good morrow, masters",
    "I am a wise fellow",
    "You are to comprehend",
    "The watch shall",
    "What is your",
]

print("\n" + "=" * 70)
print("DOGBERRY SPEAKS!")
print("=" * 70)

for temp in [0.7, 0.9]:
    print(f"\n{'='*70}")
    print(f"Temperature: {temp}")
    print(f"{'='*70}\n")

    for seed in test_seeds:
        print(f"\n🎭 Seed: \"{seed}\"")
        print("-" * 70)
        generated = model_obj.generate_text(seed, length=150, temperature=temp)
        # Show only the generated part (after seed)
        print(generated[len(seed):])
        print()

print("\n" + "=" * 70)
print("Test complete! See dogberry-model/README.md for deployment.")
print("=" * 70)
