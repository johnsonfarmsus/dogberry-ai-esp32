"""
Micro Shakespeare Language Model Trainer
Designed for ESP32-S3R8 with 8MB PSRAM constraints
Target: <1MB model size after quantization

Supports training on:
- Full Shakespeare corpus
- Character-specific dialogue (e.g., Dogberry for malapropisms)
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

class MicroShakespeareModel:
    def __init__(self, seq_length=40):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.model = None

    def extract_character_dialogue(self, text, character_name):
        """Extract dialogue for a specific character"""
        print(f"Extracting dialogue for {character_name}...")

        lines = []
        in_dialogue = False
        current_speaker = None

        # Pattern to match character names (usually in caps at start of line)
        char_pattern = re.compile(r'^([A-Z][A-Z\s]+)[\.:]\s*(.+)$', re.MULTILINE)

        # Also try to find stage directions with character speaking
        for line in text.split('\n'):
            # Check if this is a character name line
            match = char_pattern.match(line.strip())
            if match:
                speaker = match.group(1).strip()
                dialogue = match.group(2).strip()

                if character_name.upper() in speaker.upper():
                    lines.append(dialogue)
                    current_speaker = speaker
                else:
                    current_speaker = speaker
            elif current_speaker and character_name.upper() in current_speaker.upper():
                # Continuation of current character's speech
                if line.strip() and not line.strip().startswith('['):
                    lines.append(line.strip())

        extracted_text = ' '.join(lines)
        print(f"  Found {len(lines)} dialogue lines")
        print(f"  Total characters: {len(extracted_text):,}")

        if len(extracted_text) < 10000:
            print(f"  ⚠️  WARNING: Very little dialogue found for {character_name}")
            print(f"  Consider using full corpus or different character")

        return extracted_text if extracted_text else text

    def load_shakespeare_corpus(self, filepath, character_filter=None):
        """Load and prepare Shakespeare text"""
        print(f"Loading corpus from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Filter by character if specified
        if character_filter:
            text = self.extract_character_dialogue(text, character_filter)

        # Create character vocabulary
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"Corpus size: {len(text):,} characters")
        print(f"Vocabulary size: {self.vocab_size} unique characters")
        print(f"Characters: {chars[:50]}...")

        return text

    def prepare_training_data(self, text, validation_split=0.1):
        """Convert text to training sequences"""
        print(f"\nPreparing sequences (length={self.seq_length})...")

        # Encode text
        encoded = np.array([self.char_to_idx[ch] for ch in text])

        # Create sequences
        sequences = []
        next_chars = []

        step = 3  # Stride for creating sequences (smaller = more data, slower training)
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
        """
        Build ultra-compact LSTM model for ESP32

        Parameters:
        - embedding_dim: Character embedding size (32 is tiny but functional)
        - lstm_units: LSTM hidden units (128 keeps us under 500K params)
        """
        print(f"\nBuilding micro model...")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  LSTM units: {lstm_units}")

        model = keras.Sequential([
            # Embedding layer: vocab_size -> embedding_dim
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=embedding_dim,
                input_length=self.seq_length
            ),

            # Single LSTM layer (compact but effective)
            layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),

            # Output layer
            layers.Dense(self.vocab_size, activation='softmax')
        ])

        # Compile first
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Build the model to initialize parameters
        model.build(input_shape=(None, self.seq_length))

        # Now count parameters
        total_params = model.count_params()
        estimated_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        estimated_quantized_mb = (total_params * 1) / (1024 * 1024)  # 1 byte per int8

        print(f"\nModel architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Estimated size (float32): {estimated_size_mb:.2f} MB")
        print(f"  Estimated size (int8): {estimated_quantized_mb:.2f} MB")

        if estimated_quantized_mb > MAX_MODEL_SIZE_MB:
            print(f"  ⚠️  WARNING: Model may be too large for ESP32!")
        else:
            print(f"  ✓ Model should fit in ESP32 memory")

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """Train the model"""
        print(f"\nTraining for {epochs} epochs...")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'shakespeare_model_best.keras',
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
            # Pad seed text if too short
            seed_text = ' ' * (self.seq_length - len(seed_text)) + seed_text

        # Use last seq_length characters
        seed_text = seed_text[-self.seq_length:]
        generated = seed_text

        for _ in range(length):
            # Encode current sequence
            x = np.array([[self.char_to_idx.get(ch, 0) for ch in generated[-self.seq_length:]]])

            # Predict next character
            predictions = self.model.predict(x, verbose=0)[0]

            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            # Sample next character
            next_idx = np.random.choice(len(predictions), p=predictions)
            next_char = self.idx_to_char[next_idx]

            generated += next_char

        return generated

    def save_for_esp32(self, output_dir='esp32_model'):
        """Save model and metadata for ESP32 deployment"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving model for ESP32 deployment...")

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

        # Convert to TensorFlow Lite with quantization
        print("Converting to TensorFlow Lite (quantized)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Full integer quantization for maximum compression
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]

        tflite_model = converter.convert()

        # Save TFLite model
        tflite_path = f'{output_dir}/shakespeare_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        tflite_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ TFLite model saved: {tflite_path}")
        print(f"  Size: {tflite_size_mb:.2f} MB ({len(tflite_model):,} bytes)")

        # Export weights as C header for direct embedding
        self.export_to_c_header(output_dir)

        print(f"\n✓ All files saved to {output_dir}/")
        print(f"  - shakespeare_model.keras (full model)")
        print(f"  - shakespeare_model.tflite (quantized)")
        print(f"  - vocab.json (vocabulary)")
        print(f"  - model_weights.h (C header)")

    def export_to_c_header(self, output_dir):
        """Export model weights as C header file for ESP32"""
        # For now, just create a placeholder
        # We'll implement proper weight extraction if needed
        header_path = f'{output_dir}/model_weights.h'
        with open(header_path, 'w') as f:
            f.write("// Shakespeare Model Weights\n")
            f.write("// Generated for ESP32-S3R8\n\n")
            f.write(f"#define VOCAB_SIZE {self.vocab_size}\n")
            f.write(f"#define SEQ_LENGTH {self.seq_length}\n")
            f.write("\n// Load from TFLite model file instead\n")

        print(f"✓ C header saved: {header_path}")


def train_model(character_name=None, output_dir='esp32_model', epochs=30):
    """
    Train a model on Shakespeare corpus

    Args:
        character_name: Optional character to filter (e.g., "Dogberry", "Hamlet")
        output_dir: Where to save the model
        epochs: Number of training epochs
    """
    model_type = f"{character_name} personality" if character_name else "General Shakespeare"
    print("=" * 60)
    print(f"MICRO SHAKESPEARE MODEL TRAINING")
    print(f"Mode: {model_type}")
    print("Target: ESP32-S3R8 (8MB PSRAM, 16MB Flash)")
    print("=" * 60)

    # Initialize model
    model = MicroShakespeareModel(seq_length=40)

    # Load corpus
    corpus_path = 'shakespeare/shakespeare_100_complete_works_of_william_shakespeare.txt'
    text = model.load_shakespeare_corpus(corpus_path, character_filter=character_name)

    # Prepare data
    (X_train, y_train), (X_val, y_val) = model.prepare_training_data(text)

    # Build model
    model.build_model(embedding_dim=32, lstm_units=128)

    # Show model summary
    print("\n" + "=" * 60)
    model.model.summary()
    print("=" * 60)

    # Train
    history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=256)

    # Test generation
    print("\n" + "=" * 60)
    print("SAMPLE GENERATION")
    print("=" * 60)

    if character_name:
        test_seeds = [
            "Good morrow",
            "I am a wise",
            "The watch has"
        ]
    else:
        test_seeds = [
            "To be, or not to be",
            "O Romeo, Romeo",
            "All the world's a stage"
        ]

    for seed in test_seeds:
        print(f"\nSeed: '{seed}'")
        print("-" * 60)
        generated = model.generate_text(seed, length=200, temperature=0.8)
        print(generated)

    # Save for ESP32
    model.save_for_esp32(output_dir)

    print("\n" + "=" * 60)
    print(f"✓ TRAINING COMPLETE!")
    print(f"Model saved to: {output_dir}/")
    print("=" * 60)

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Shakespeare micro-model for ESP32')
    parser.add_argument('--character', type=str, default=None,
                       help='Character name to train on (e.g., "Dogberry", "Hamlet")')
    parser.add_argument('--output', type=str, default='esp32_model',
                       help='Output directory for model files')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--all', action='store_true',
                       help='Train both general and Dogberry models')

    args = parser.parse_args()

    if args.all:
        # Train both models
        print("\n" + "🎭" * 30)
        print("TRAINING BOTH MODELS")
        print("🎭" * 30 + "\n")

        # General model
        print("\n1️⃣  Training GENERAL Shakespeare model...")
        train_model(character_name=None, output_dir='esp32_model_general', epochs=args.epochs)

        print("\n\n")

        # Dogberry model
        print("\n2️⃣  Training DOGBERRY personality model...")
        train_model(character_name='Dogberry', output_dir='esp32_model_dogberry', epochs=args.epochs)

        print("\n" + "=" * 60)
        print("✓ BOTH MODELS TRAINED!")
        print("  - esp32_model_general/    (General Shakespeare)")
        print("  - esp32_model_dogberry/   (Dogberry personality)")
        print("=" * 60)
    else:
        # Train single model
        train_model(character_name=args.character, output_dir=args.output, epochs=args.epochs)

    print("\nNext steps:")
    print("1. Test the model with different temperatures")
    print("2. Copy esp32_model/shakespeare_model.tflite to ESP32")
    print("3. Copy esp32_model/vocab.json to ESP32")
    print("4. Upload to SPIFFS and flash the sketch")


if __name__ == '__main__':
    main()
