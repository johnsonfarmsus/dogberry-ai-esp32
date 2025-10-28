"""
Shakespeare Micro-Model Trainer
Train on FULL Shakespeare corpus for quality,
then use Dogberry-style prompting in the bot

Target: ESP32-S3R8 with 8MB PSRAM
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

class ShakespeareModel:
    def __init__(self, seq_length=40):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.model = None

    def load_corpus(self, filepath):
        """Load full Shakespeare corpus"""
        print(f"Loading corpus from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Create character vocabulary
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"Corpus size: {len(text):,} characters")
        print(f"Vocabulary size: {self.vocab_size} unique characters")

        return text

    def prepare_training_data(self, text, validation_split=0.1):
        """Convert text to training sequences"""
        print(f"\nPreparing sequences (length={self.seq_length})...")

        # Encode text
        encoded = np.array([self.char_to_idx[ch] for ch in text])

        # Create sequences
        sequences = []
        next_chars = []

        step = 3  # Stride
        for i in range(0, len(encoded) - self.seq_length, step):
            sequences.append(encoded[i:i + self.seq_length])
            next_chars.append(encoded[i + self.seq_length])

        X = np.array(sequences)
        y = np.array(next_chars)

        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training sequences: {len(X_train):,}")
        print(f"Validation sequences: {len(X_val):,}")

        return (X_train, y_train), (X_val, y_val)

    def build_model(self, embedding_dim=32, lstm_units=128):
        """Build micro-LSTM for ESP32"""
        print(f"\nBuilding Shakespeare model...")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  LSTM units: {lstm_units}")

        model = keras.Sequential([
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=embedding_dim
            ),
            layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(self.vocab_size, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.build(input_shape=(None, self.seq_length))

        total_params = model.count_params()
        estimated_size_mb = (total_params * 4) / (1024 * 1024)

        print(f"\nModel architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Estimated size (float32): {estimated_size_mb:.2f} MB")

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
        """Train the model"""
        print(f"\nTraining for {epochs} epochs...")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'shakespeare_best.keras',
                save_best_only=True,
                monitor='val_loss'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

    def generate_text(self, seed_text, length=200, temperature=1.0):
        """Generate text"""
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

    def save_for_esp32(self, output_dir='model'):
        """Save model for ESP32"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving model...")

        # Save full model
        self.model.save(f'{output_dir}/shakespeare_model.keras')

        # Save vocabulary
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()},
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length
        }

        with open(f'{output_dir}/vocab.json', 'w') as f:
            json.dump(vocab_data, f, indent=2)

        # Export to TFLite
        print("Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        tflite_path = f'{output_dir}/shakespeare_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"✓ TFLite model: {tflite_path} ({size_kb:.1f} KB)")

        print(f"\n✓ All files saved to {output_dir}/")


def main():
    print("=" * 70)
    print("SHAKESPEARE MICRO-MODEL TRAINING")
    print("Full corpus for quality text generation")
    print("Use Dogberry-style prompting in bot for personality")
    print("=" * 70)

    model = ShakespeareModel(seq_length=40)

    # Load FULL corpus
    corpus_path = 'shakespeare/shakespeare_100_complete_works_of_william_shakespeare.txt'
    text = model.load_corpus(corpus_path)

    # Prepare data
    (X_train, y_train), (X_val, y_val) = model.prepare_training_data(text)

    # Build model
    model.build_model(embedding_dim=32, lstm_units=128)

    print("\n" + "=" * 70)
    model.model.summary()
    print("=" * 70)

    # Train
    history = model.train(X_train, y_train, X_val, y_val, epochs=40, batch_size=256)

    # Test generation
    print("\n" + "=" * 70)
    print("SAMPLE GENERATION")
    print("=" * 70)

    test_seeds = [
        "To be, or not to be",
        "Good morrow",
        "What say you"
    ]

    for seed in test_seeds:
        print(f"\nSeed: '{seed}'")
        print("-" * 70)
        generated = model.generate_text(seed, length=200, temperature=0.8)
        print(generated)

    # Save
    model.save_for_esp32('model')

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("\nDogberry Bot Strategy:")
    print("1. Model generates quality Shakespeare text")
    print("2. Bot prepends Dogberry-style phrases:")
    print("   - 'Good morrow, friend! Verily, I say...'")
    print("   - 'As a most wise fellow, I comprehend...'")
    print("   - 'The watch doth bid me tell you...'")
    print("3. ESP32 selects responses with Dogberry keywords")
    print("=" * 70)


if __name__ == '__main__':
    main()
