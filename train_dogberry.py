"""
Dogberry Character Model Trainer
Train a micro-LLM on Dogberry's dialogue for ESP32-S3R8

Dogberry is the comic constable from "Much Ado About Nothing"
Famous for malapropisms like:
- "comprehend all vagrom men" (apprehend vagrant men)
- "most tolerable and not to be endured"
- "desartless man" (deserving man)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
import re
from pathlib import Path

# ESP32 Memory Constraints
MAX_MODEL_SIZE_MB = 1.0  # Target size after quantization
PSRAM_MB = 8
TARGET_PARAMS = 500_000  # Aim for ~500K parameters

class DogberryModel:
    def __init__(self, seq_length=40):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.model = None

    def extract_dogberry_dialogue(self, text):
        """Extract Dogberry's dialogue from Shakespeare corpus"""
        print(f"Extracting Dogberry's dialogue...")

        lines = []
        is_dogberry = False

        for line in text.split('\n'):
            line = line.strip()

            # Check if this line is a character name
            if line and line.isupper() and '.' in line:
                # This is a character name line
                if 'DOGBERRY' in line:
                    is_dogberry = True
                else:
                    is_dogberry = False
            elif is_dogberry and line:
                # This is Dogberry's dialogue
                # Skip stage directions
                if not line.startswith('[') and not line.endswith(']'):
                    lines.append(line)

        extracted_text = '\n'.join(lines)
        print(f"  Found {len(lines)} dialogue lines")
        print(f"  Total characters: {len(extracted_text):,}")

        if len(extracted_text) < 10000:
            print(f"  ⚠️  WARNING: Limited dialogue found")
            print(f"  Note: Small dataset may need more epochs for good results")

        # Show sample
        print(f"\nSample Dogberry dialogue:")
        sample = extracted_text[:200] if len(extracted_text) > 200 else extracted_text
        print(f"  \"{sample}...\"")

        return extracted_text

    def load_corpus(self, filepath):
        """Load and prepare Dogberry's dialogue"""
        print(f"Loading corpus from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract Dogberry's lines
        text = self.extract_dogberry_dialogue(text)

        # Create character vocabulary
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"\nVocabulary size: {self.vocab_size} unique characters")

        return text

    def prepare_training_data(self, text, validation_split=0.1):
        """Convert text to training sequences"""
        print(f"\nPreparing sequences (length={self.seq_length})...")

        # Encode text
        encoded = np.array([self.char_to_idx[ch] for ch in text])

        # Create sequences
        sequences = []
        next_chars = []

        step = 3  # Stride for creating sequences
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
        """Build ultra-compact LSTM model for ESP32"""
        print(f"\nBuilding Dogberry model...")
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

        # Build to initialize parameters
        model.build(input_shape=(None, self.seq_length))

        # Count parameters
        total_params = model.count_params()
        estimated_size_mb = (total_params * 4) / (1024 * 1024)
        estimated_quantized_mb = (total_params * 1) / (1024 * 1024)

        print(f"\nModel architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Estimated size (float32): {estimated_size_mb:.2f} MB")
        print(f"  Estimated size (int8): {estimated_quantized_mb:.2f} MB")

        if estimated_quantized_mb > MAX_MODEL_SIZE_MB:
            print(f"  ⚠️  WARNING: Model may be too large for ESP32!")
        else:
            print(f"  ✓ Model fits in ESP32 memory!")

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """Train the model"""
        print(f"\nTraining for {epochs} epochs...")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'dogberry_model_best.keras',
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
        """Generate text from the model"""
        if len(seed_text) < self.seq_length:
            seed_text = ' ' * (self.seq_length - len(seed_text)) + seed_text

        seed_text = seed_text[-self.seq_length:]
        generated = seed_text

        for _ in range(length):
            x = np.array([[self.char_to_idx.get(ch, 0) for ch in generated[-self.seq_length:]]])
            predictions = self.model.predict(x, verbose=0)[0]

            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            next_idx = np.random.choice(len(predictions), p=predictions)
            next_char = self.idx_to_char[next_idx]

            generated += next_char

        return generated

    def save_for_esp32(self, output_dir='model'):
        """Save model and metadata for ESP32 deployment"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving model for ESP32...")

        # Save full model
        self.model.save(f'{output_dir}/dogberry_model.keras')

        # Save vocabulary
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()},
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length
        }

        with open(f'{output_dir}/vocab.json', 'w') as f:
            json.dump(vocab_data, f, indent=2)

        # Convert to TensorFlow Lite with quantization
        print("Converting to TensorFlow Lite (quantized)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Use dynamic range quantization (simpler, no representative dataset needed)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # For ESP32, we want the smallest model possible
        # Dynamic range quantization reduces size significantly
        tflite_model = converter.convert()

        tflite_path = f'{output_dir}/dogberry_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        tflite_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ TFLite model saved: {tflite_path}")
        print(f"  Size: {tflite_size_mb:.2f} MB ({len(tflite_model):,} bytes)")

        print(f"\n✓ All files saved to {output_dir}/")
        print(f"  - dogberry_model.keras (full model)")
        print(f"  - dogberry_model.tflite (quantized for ESP32)")
        print(f"  - vocab.json (character mappings)")


def main():
    print("=" * 70)
    print("DOGBERRY CHARACTER MODEL TRAINING")
    print("Comic Constable from Much Ado About Nothing")
    print("Target: ESP32-S3R8 (8MB PSRAM, 16MB Flash)")
    print("=" * 70)

    # Initialize model
    model = DogberryModel(seq_length=40)

    # Load corpus
    corpus_path = 'shakespeare/shakespeare_100_complete_works_of_william_shakespeare.txt'
    text = model.load_corpus(corpus_path)

    # Prepare data
    (X_train, y_train), (X_val, y_val) = model.prepare_training_data(text)

    # Build model
    model.build_model(embedding_dim=32, lstm_units=128)

    # Show model summary
    print("\n" + "=" * 70)
    model.model.summary()
    print("=" * 70)

    # Train
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=256)

    # Test generation
    print("\n" + "=" * 70)
    print("DOGBERRY SAMPLE GENERATION")
    print("=" * 70)

    test_seeds = [
        "Good morrow, masters",
        "You are to comprehend",
        "I am a wise fellow"
    ]

    for seed in test_seeds:
        print(f"\nSeed: '{seed}'")
        print("-" * 70)
        generated = model.generate_text(seed, length=200, temperature=0.8)
        print(generated)
        print()

    # Save for ESP32
    model.save_for_esp32('model')

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("\nNext steps:")
    print("1. Copy model/ folder to ../dogberry-bot/")
    print("2. Test generation with different temperatures")
    print("3. Deploy to ESP32 via SPIFFS")
    print("=" * 70)


if __name__ == '__main__':
    main()
