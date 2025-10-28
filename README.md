# Dogberry Model - Pico Language Model for ESP32

A training pipeline for creating tiny LSTM language models that run on ESP32 microcontrollers. This project demonstrates how to train, optimize, and deploy a 1.6M parameter word-level model that fits in 6.15MB and generates Shakespearean-style text on-device.

## Overview

This repository contains everything needed to train a "Pico Language Model" (PicoLM) - a highly constrained neural network designed to run on resource-limited embedded devices while still producing coherent, contextual text generation.

### Key Achievements

- **6.15MB model** running on ESP32-S3 with 8MB PSRAM
- **Word-level tokenization** (4000 vocab) for better coherence than character-level
- **Contextual generation** that responds appropriately to user input
- **47% flash usage** on 16MB device with custom partition table
- **~2-3 second** inference time per response

## Model Architecture

### Specifications

```
Input: 40-word context window
├── Embedding Layer (4000 vocab → 64 dim)
├── LSTM Layer (256 units)
└── Dense Output (256 → 4000 vocab)

Total Parameters: 1,612,704
Size (float32): 6.15 MB
Training Data: Shakespeare's complete works
```

### Why This Architecture?

**Word-Level vs Character-Level**:
- Character models are smaller but produce gibberish
- Word models are larger but generate coherent sentences
- 4000-word vocabulary covers 95%+ of Shakespeare

**LSTM Size Trade-offs**:
- 256 units: Good balance of quality and size
- 128 units: 50% smaller, noticeably worse quality
- 512 units: Better quality, but 13MB+ (too large for ESP32)

**Context Window**:
- 40 words provides enough context for conversation
- Longer contexts exponentially increase memory usage
- Shorter contexts produce less coherent responses

## Requirements

### Software

```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow 2.x
- NumPy
- Keras

### Hardware (for deployment)

- ESP32-S3 with 8MB PSRAM
- 16MB Flash minimum
- See [dogberry-bot](https://github.com/yourusername/dogberry-bot) for implementation

## Training Pipeline

### 1. Prepare Training Data

The model is trained on Shakespeare's complete works:

```python
# Training corpus includes:
- All 38 plays
- 154 sonnets
- Narrative poems
- Total: ~5MB of text, ~900K words
```

Located in `shakespeare/` directory.

### 2. Train the Model

```bash
python3 train_word_level_production.py
```

**Training configuration**:
```python
VOCAB_SIZE = 4000        # Top 4K most frequent words
SEQ_LENGTH = 40          # 40-word context
EMBEDDING_DIM = 64       # Embedding dimension
LSTM_UNITS = 256         # LSTM hidden units
BATCH_SIZE = 128
EPOCHS = 60              # Typically 40-60 epochs
```

**Expected training time**:
- CPU: ~8-12 hours
- GPU: ~2-4 hours

**Training output**:
```
model/
├── dogberry_word_production_best.keras     # Best model checkpoint
├── vocab_word_level_production.json        # Vocabulary mapping
└── training_history.json                   # Loss/accuracy curves
```

### 3. Export for ESP32

```bash
python3 export_to_esp32.py
```

This generates two C header files:
- `model_weights_word.h` (~22MB source, 6.15MB binary)
- `vocab_data_word.h` (~89KB)

**Key export features**:
- Adds `PROGMEM` attribute to keep weights in flash
- Exports float32 (no quantization needed)
- Formats arrays for ESP32 memory access patterns

### 4. Deploy to ESP32

Copy the generated headers to the [dogberry-bot](https://github.com/yourusername/dogberry-bot) repository:

```bash
cp model_weights_word.h ../dogberry-bot/esp32_firmware/src/
cp vocab_data_word.h ../dogberry-bot/esp32_firmware/src/
```

## Training Scripts

### Production Training

**`train_word_level_production.py`** - Main training script
- 60 epochs with early stopping
- Word-level tokenization
- Model checkpointing
- Validation monitoring

### Export Utilities

**`export_to_esp32.py`** - C header generation
- Converts Keras model to C arrays
- Adds PROGMEM attributes
- Generates vocabulary mapping

## Memory Optimization Strategies

### Problem: Model Too Large for ESP32

The word-level model presents unique challenges:

| Component | Size | Challenge |
|-----------|------|-----------|
| Embedding | 1.0 MB | 4000 × 64 floats |
| Dense Layer | **4.1 MB** | 256 × 4000 floats (biggest!) |
| LSTM Weights | 1.0 MB | Recurrent matrices |
| **Total** | **6.15 MB** | Must fit in 8MB PSRAM |

### Solutions Implemented

**1. PROGMEM Storage**
```c
const float EMBEDDING_WEIGHTS[256000] PROGMEM = {...};
```
- Keeps weights in flash, not RAM
- Read on-demand with `pgm_read_float()`
- Saves 6MB of RAM

**2. Static Buffer Allocation**
```c
// Pre-allocate at startup, reuse for all inferences
float* logits = (float*)ps_malloc(VOCAB_SIZE * sizeof(float));
```
- Avoids malloc/free during generation
- Prevents memory fragmentation
- Uses PSRAM for large buffers

**3. Custom Partition Table**
```csv
app0, app, factory, 0x10000, 0xF00000  # 15MB app partition
```
- Removed OTA support (don't need dual partitions)
- Expanded from 10MB to 15MB
- Firmware uses 47% (7.4MB / 15MB)

### Alternatives Considered

**Option 1: Smaller Vocabulary (2K words)**
- Pros: Reduces dense layer to 2MB
- Cons: More <UNK> tokens, worse quality
- Decision: Rejected - quality matters more

**Option 2: Quantization (INT8)**
- Pros: 75% size reduction
- Cons: Complex implementation, accuracy loss
- Decision: Deferred - float32 works

**Option 3: Smaller LSTM (128 units)**
- Pros: Halves LSTM and dense layer
- Cons: Noticeably worse generation quality
- Decision: Rejected - 256 is minimum for good output

## Design Decisions

### Why Word-Level?

**Character-level output** (actual example):
```
"nay i am an ho  neft watfchin the townd  "
```

**Word-level output** (actual example):
```
"nay sir I am no villain but an honest watchman of the town"
```

The improvement in coherence justified the 4x model size increase.

### Why 4000-Word Vocabulary?

**Coverage analysis**:
- 1000 words: 85% coverage, many <UNK>
- 2000 words: 92% coverage, frequent <UNK>
- **4000 words: 95%+ coverage, rare <UNK>** ✓
- 8000 words: 97% coverage, model too large

4000 words provides the best quality/size trade-off.

### Why 60 Epochs?

**Training curve analysis**:
- Epochs 1-20: Rapid loss decrease
- Epochs 20-40: Steady improvement
- Epochs 40-60: Diminishing returns
- **Epochs 60+: Overfitting risk**

60 epochs with early stopping provides best generalization.

## Troubleshooting

### Model Won't Fit on ESP32

**Symptom**: Bootloop or allocation failures

**Solutions**:
1. Verify PSRAM enabled: `-DBOARD_HAS_PSRAM`
2. Check partition table: 15MB app partition
3. Confirm PROGMEM in weights: `const float X[] PROGMEM`
4. Use static allocation, not malloc during inference

### Poor Generation Quality

**Symptom**: Repetitive or incoherent output

**Solutions**:
1. Train longer (try 80 epochs)
2. Increase temperature (try 0.9 or 1.0)
3. Verify training data quality
4. Check if model converged (loss curve)

### Slow Inference

**Symptom**: >10 seconds per response

**Possible causes**:
1. Not using PROGMEM (copying weights to RAM)
2. Using malloc/free in generation loop
3. Reading weights inefficiently
4. CPU clock too low

**Expected**: 2-3 seconds for 40-word generation

## File Structure

```
dogberry-model/
├── train_word_level_production.py   # Main training script
├── export_to_esp32.py               # Export to C headers
├── requirements.txt                 # Python dependencies
├── shakespeare/                     # Training corpus
│   └── shakespeare_*.txt            # Individual works
├── model/                           # Training outputs (git-ignored)
│   ├── dogberry_word_production_best.keras
│   ├── vocab_word_level_production.json
│   └── export_summary.txt
└── README.md
```

## Performance Benchmarks

### Training

| Configuration | Time | Final Loss | Quality |
|--------------|------|------------|---------|
| 40 epochs, CPU | 6hrs | 2.8 | Good |
| 60 epochs, CPU | 10hrs | 2.4 | Better |
| 60 epochs, GPU | 3hrs | 2.4 | Better |

### Inference (ESP32-S3 @ 240MHz)

| Operation | Time | Memory |
|-----------|------|--------|
| Model load | 2s | 33KB PSRAM |
| Single word | 50ms | (reuses buffers) |
| 40-word response | 2-3s | (reuses buffers) |

### Model Sizes

| Variant | Vocab | LSTM | Params | Size | Quality |
|---------|-------|------|--------|------|---------|
| Char-level | 52 | 128 | 100K | 0.4MB | Poor |
| Word-1K | 1000 | 128 | 500K | 2MB | Fair |
| **Word-4K** | **4000** | **256** | **1.6M** | **6.15MB** | **Good** ✓ |
| Word-8K | 8000 | 256 | 3.2M | 12.8MB | Excellent (too large) |

## License

MIT License - See LICENSE file

Training data (Shakespeare) is public domain.

## Related Projects

- [dogberry-bot](https://github.com/yourusername/dogberry-bot) - Bluesky bot implementation

## Contributing

Contributions welcome! Areas of interest:
- Quantization implementation
- Alternative architectures (Transformer, GRU)
- Training optimizations
- Deployment to other platforms
