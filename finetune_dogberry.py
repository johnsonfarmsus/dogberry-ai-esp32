"""
Fine-tune Shakespeare Model with Dogberry Personality

Takes the base Shakespeare model and continues training
on Dogberry-heavy corpus to inject personality.

Strategy:
- Load pre-trained Shakespeare model
- Continue training on mixed corpus (95% Shakespeare + 5% Dogberry)
- Use lower learning rate to avoid catastrophic forgetting
- Train for 10-15 epochs
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

class DogberryFineTuner:
    def __init__(self):
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.seq_length = 40

    def load_base_model(self, model_path='model/shakespeare_model.keras'):
        """Load the pre-trained Shakespeare model"""
        print(f"Loading base model from {model_path}...")

        self.model = keras.models.load_model(model_path)

        # Load vocabulary
        with open('model/vocab.json', 'r') as f:
            vocab = json.load(f)
            self.char_to_idx = vocab['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
            self.vocab_size = vocab['vocab_size']
            self.seq_length = vocab['seq_length']

        print(f"✓ Base model loaded")
        print(f"  Parameters: {self.model.count_params():,}")
        return True

    def prepare_finetuning_corpus(self):
        """Load and mix Shakespeare + Dogberry corpus"""
        print("\nPreparing fine-tuning corpus...")

        # Load original Shakespeare
        with open('shakespeare/shakespeare_100_complete_works_of_william_shakespeare.txt', 'r') as f:
            shakespeare = f.read()

        # Load Dogberry corpus (real + synthetic)
        try:
            with open('dogberry_corpus_combined.txt', 'r') as f:
                dogberry = f.read()
        except FileNotFoundError:
            print("⚠️  No dogberry_corpus_combined.txt found")
            print("   Run prepare_finetuning_data.py first")
            return None

        # Mix corpus - repeat Dogberry to balance
        # We want ~5% of training to be Dogberry-focused
        dogberry_repeats = int(len(shakespeare) * 0.05 / len(dogberry))
        dogberry_repeated = (dogberry + '\n\n') * dogberry_repeats

        # Combine
        combined = shakespeare + '\n\n' + dogberry_repeated

        print(f"  Shakespeare: {len(shakespeare):,} chars")
        print(f"  Dogberry (original): {len(dogberry):,} chars")
        print(f"  Dogberry (repeated {dogberry_repeats}x): {len(dogberry_repeated):,} chars")
        print(f"  Combined: {len(combined):,} chars")
        print(f"  Dogberry percentage: {len(dogberry_repeated)/len(combined)*100:.1f}%")

        return combined

    def prepare_training_data(self, text, validation_split=0.1):
        """Convert text to training sequences"""
        print(f"\nPreparing sequences...")

        # Encode text
        encoded = np.array([self.char_to_idx[ch] for ch in text if ch in self.char_to_idx])

        # Create sequences
        sequences = []
        next_chars = []

        step = 3
        for i in range(0, len(encoded) - self.seq_length, step):
            sequences.append(encoded[i:i + self.seq_length])
            next_chars.append(encoded[i + self.seq_length])

        X = np.array(sequences)
        y = np.array(next_chars)

        # Split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"  Training sequences: {len(X_train):,}")
        print(f"  Validation sequences: {len(X_val):,}")

        return (X_train, y_train), (X_val, y_val)

    def finetune(self, X_train, y_train, X_val, y_val, epochs=10):
        """Fine-tune the model with lower learning rate"""
        print(f"\nFine-tuning for {epochs} epochs...")
        print("Using low learning rate to preserve Shakespeare knowledge...")

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 10x lower than initial
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'shakespeare_dogberry_finetuned.keras',
                save_best_only=True,
                monitor='val_loss'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.00001
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            batch_size=256,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

    def generate_text(self, seed_text, length=200, temperature=0.8):
        """Test text generation"""
        if len(seed_text) < self.seq_length:
            seed_text = ' ' * (self.seq_length - len(seed_text)) + seed_text

        seed_text = seed_text[-self.seq_length:]
        generated = seed_text

        for _ in range(length):
            x = np.array([[self.char_to_idx.get(ch, 0) for ch in generated[-self.seq_length:]]])
            predictions = self.model.predict(x, verbose=0)[0]

            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            next_idx = np.random.choice(len(predictions), p=predictions)
            next_char = self.idx_to_char[next_idx]

            generated += next_char

        return generated

    def save_finetuned_model(self, output_dir='model'):
        """Save the fine-tuned model"""
        print(f"\nSaving fine-tuned model...")

        # Save full model
        self.model.save(f'{output_dir}/shakespeare_dogberry_finetuned.keras')
        print(f"✓ Saved: {output_dir}/shakespeare_dogberry_finetuned.keras")

        # Vocab stays the same
        print(f"✓ Using existing vocab.json")


def main():
    print("=" * 70)
    print("DOGBERRY FINE-TUNING")
    print("Inject personality into Shakespeare model")
    print("=" * 70)

    tuner = DogberryFineTuner()

    # Load base model
    if not tuner.load_base_model():
        print("ERROR: Could not load base model")
        print("Make sure base training finished and model exists")
        return

    # Prepare corpus
    text = tuner.prepare_finetuning_corpus()
    if not text:
        return

    # Prepare data
    (X_train, y_train), (X_val, y_val) = tuner.prepare_training_data(text)

    # Fine-tune
    history = tuner.finetune(X_train, y_train, X_val, y_val, epochs=10)

    # Test generation
    print("\n" + "=" * 70)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 70)

    test_seeds = [
        "Good morrow",
        "I am a wise",
        "The watch shall",
        "What say you",
    ]

    for seed in test_seeds:
        print(f"\nSeed: '{seed}'")
        print("-" * 70)
        generated = tuner.generate_text(seed, length=200, temperature=0.8)
        print(generated)

    # Save
    tuner.save_finetuned_model()

    print("\n" + "=" * 70)
    print("✓ FINE-TUNING COMPLETE!")
    print("\nNext steps:")
    print("1. Test generation quality")
    print("2. Export to TFLite: python3 export_tflite.py")
    print("3. Deploy to ESP32")
    print("=" * 70)


if __name__ == '__main__':
    main()
