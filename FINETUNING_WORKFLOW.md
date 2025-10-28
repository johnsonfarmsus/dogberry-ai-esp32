# Dogberry Fine-Tuning Workflow

Complete step-by-step guide for training a Shakespeare model and injecting Dogberry personality through fine-tuning.

## Overview

**Strategy**: Two-stage training approach
1. **Stage 1**: Train base model on full Shakespeare corpus (5.3MB) for quality text generation
2. **Stage 2**: Fine-tune on Dogberry-heavy corpus (5% mix) to inject personality without losing Shakespeare quality

## Why Fine-Tuning?

**Problem**: Training only on Dogberry dialogue (~500 lines) produces gibberish
- Not enough data to learn language patterns
- Model overfits to limited vocabulary
- Output is incoherent

**Solution**: Train on full Shakespeare, then fine-tune with Dogberry
- Base model learns quality Shakespearean English
- Fine-tuning injects Dogberry personality without catastrophic forgetting
- Lower learning rate preserves base knowledge while adding character traits

## Complete Workflow

### Step 1: Base Training (Currently Running)

Train on full Shakespeare corpus to get quality text generation:

```bash
cd dogberry-model
python3 train_shakespeare.py
```

**What it does**:
- Loads 5.3MB Shakespeare corpus (100 complete works)
- Trains character-level LSTM for 40 epochs
- Saves `model/shakespeare_model.keras` (base model)
- Exports `model/shakespeare_model.tflite` (121KB for ESP32)

**Duration**: ~6 hours (40 epochs)

**Monitor progress**:
```bash
# Check process
ps aux | grep train_shakespeare

# Watch CPU usage (should be ~400-500%)
top -pid <process_id>

# Model will auto-save when done
ls -lh model/shakespeare_model.keras
```

**Expected output**:
- Validation loss: ~1.5-1.8
- Validation accuracy: ~50-55%
- Quality: Coherent Shakespearean text

### Step 2: Generate Synthetic Dogberry Corpus

While base training runs, create synthetic Dogberry training data:

```bash
# Extract real Dogberry dialogue from Shakespeare
python3 generate_dogberry_corpus.py
```

**Output**: `DOGBERRY_GENERATION_GUIDE.md`

**Manual step** (use Claude/GPT-4):

Open `DOGBERRY_GENERATION_GUIDE.md` and use the 8 prompt categories to generate synthetic Dogberry text:

1. **Constable duties** - Dogberry giving watch instructions
2. **Character interactions** - Dialogues with other characters
3. **Observations** - Commentary on events
4. **Teachings** - Dogberry's wisdom and advice
5. **Proclamations** - Official announcements
6. **Investigations** - Interrogations and examinations
7. **Personal reflections** - Soliloquy-style thoughts
8. **Reactions** - Responses to various situations

**Target**: 50-100KB total text (real + synthetic)

**Example prompt**:
```
Write 20 lines of dialogue where Dogberry is instructing the night watch
on their duties. Use his characteristic malapropisms:
- "comprehend" instead of "apprehend"
- "vagrom" instead of "vagrant"
- "desartless" instead of "deserving"
Keep the Elizabethan style but inject his bumbling, self-important tone.
```

Save all generated text to: `dogberry_synthetic_corpus.txt`

### Step 3: Prepare Fine-Tuning Data

Combine real dialogue + synthetic corpus:

```bash
python3 prepare_finetuning_data.py
```

**What it does**:
- Loads `dogberry_real_dialogue.txt` (extracted from Shakespeare)
- Loads `dogberry_synthetic_corpus.txt` (your generated text)
- Combines into `dogberry_corpus_combined.txt`
- Reports statistics and corpus percentages

**Expected output**:
```
Real Dogberry dialogue: ~15,000 characters
Synthetic corpus: ~50,000 characters
Combined: ~65,000 characters
Dogberry percentage: ~1.2% of Shakespeare corpus ✓
```

### Step 4: Fine-Tune Model

Once base training completes (check `model/shakespeare_model.keras` exists):

```bash
python3 finetune_dogberry.py
```

**What it does**:
- Loads pre-trained `shakespeare_model.keras`
- Mixes training data: 95% Shakespeare + 5% Dogberry (repeated)
- Recompiles with **low learning rate** (0.0001 - 10x lower than base)
- Trains for 10 epochs with early stopping
- Tests generation quality
- Saves `shakespeare_dogberry_finetuned.keras`

**Duration**: ~1-2 hours (10 epochs)

**Why low learning rate?**
- Prevents "catastrophic forgetting" of Shakespeare knowledge
- Gently nudges model toward Dogberry personality
- Preserves quality while adding character

**Expected output**:
```
Validation loss should stay ~1.5-2.0 (close to base model)
Generation should show Dogberry traits:
- Malapropisms appear naturally
- Self-important tone
- Bumbling but earnest
```

### Step 5: Export Fine-Tuned Model

Export the fine-tuned model to TFLite for ESP32:

```bash
python3 export_tflite.py
```

**What it does**:
- Loads `shakespeare_dogberry_finetuned.keras`
- Converts to TFLite with dynamic range quantization
- Saves `shakespeare_dogberry_finetuned.tflite` (~121KB)
- Vocabulary unchanged (uses existing `vocab.json`)

### Step 6: Deploy to ESP32

Copy model files to bot:

```bash
# Copy fine-tuned model
cp model/shakespeare_dogberry_finetuned.tflite ../dogberry-bot/esp32_inference/data/shakespeare_model.tflite

# Vocab stays the same
cp model/vocab.json ../dogberry-bot/esp32_inference/data/
```

**Update Arduino sketch** (remove post-processing hacks):

Since the fine-tuned model naturally includes Dogberry personality, you can simplify the ESP32 code by removing:
- `dogberry_intros[]` array (not needed - model handles this)
- `malapropisms[]` replacements (not needed - model learned these)
- `addDogberryPersonality()` function (optional - model already has personality)

Just use `generateText()` directly!

### Step 7: Test Generation Quality

Test the fine-tuned model before deploying:

```bash
python3 test_generation.py
```

Try various seeds:
- "Good morrow" - Should generate Dogberry-style greeting
- "The watch" - Should mention constable duties
- "I am a wise" - Should show self-importance
- "What villain" - Should investigate with malapropisms

**Quality checklist**:
- ✓ Text is coherent Shakespearean English
- ✓ Malapropisms appear naturally (not forced)
- ✓ Dogberry personality comes through
- ✓ Not too repetitive
- ✓ Maintains proper grammar (mostly)

### Step 8: Deploy and Monitor

Upload to ESP32:
1. Upload model files via SPIFFS
2. Flash `dogberry_bot.ino`
3. Monitor serial output
4. Test button generation
5. Check Bluesky integration

## Troubleshooting Fine-Tuning

### Problem: Model forgets Shakespeare after fine-tuning

**Symptoms**: Output becomes incoherent, loses grammar, repetitive

**Solutions**:
- Lower learning rate even more (try 0.00005)
- Reduce Dogberry percentage (try 2-3% instead of 5%)
- Train for fewer epochs (try 5 instead of 10)
- Increase early stopping patience

### Problem: No Dogberry personality after fine-tuning

**Symptoms**: Generates Shakespeare but no malapropisms or character traits

**Solutions**:
- Increase Dogberry corpus size (100KB+)
- Ensure synthetic corpus has clear Dogberry markers
- Train longer (15-20 epochs)
- Increase Dogberry percentage to 8-10%

### Problem: Fine-tuning takes too long

**Solutions**:
- Reduce batch size (try 128 instead of 256)
- Use fewer validation sequences
- Reduce Dogberry repeats in corpus mix

### Problem: TFLite export fails

**Error**: `ValueError: For full integer quantization, a 'representative_dataset' must be specified`

**Fix**: The code already uses dynamic range quantization (no representative dataset needed). If you see this error, check that `export_tflite.py` has:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# NOT: converter.target_spec.supported_types = [tf.int8]
```

## File Reference

| File | Purpose | When |
|------|---------|------|
| `train_shakespeare.py` | Base model training | Step 1 (first) |
| `generate_dogberry_corpus.py` | Extract real dialogue + create guide | Step 2 (during base training) |
| `DOGBERRY_GENERATION_GUIDE.md` | Prompts for Claude/GPT-4 | Step 2 (manual generation) |
| `dogberry_synthetic_corpus.txt` | Your generated Dogberry text | Step 2 (create manually) |
| `prepare_finetuning_data.py` | Combine corpora | Step 3 (after generation) |
| `dogberry_corpus_combined.txt` | Final fine-tuning corpus | Step 3 (auto-created) |
| `finetune_dogberry.py` | Fine-tune base model | Step 4 (after base training) |
| `export_tflite.py` | Export to ESP32 format | Step 5 (after fine-tuning) |
| `test_generation.py` | Test quality | Step 7 (before deploy) |

## Expected Timeline

| Stage | Duration | Can Run Concurrently |
|-------|----------|----------------------|
| Base training | 6 hours | - |
| Generate corpus (manual) | 1-2 hours | ✓ (during base training) |
| Prepare data | 1 minute | - |
| Fine-tuning | 1-2 hours | - |
| Export to TFLite | 1 minute | - |
| Test generation | 5 minutes | - |
| Deploy to ESP32 | 10 minutes | - |
| **Total** | **~7-10 hours** | (with concurrent work: ~7-8 hours) |

## Theory: Why This Works

**Base Training** gives the model:
- Proper grammar and syntax
- Rich vocabulary
- Shakespearean style
- Coherent text generation

**Fine-Tuning** adds:
- Dogberry vocabulary preferences (malapropisms)
- Character personality patterns
- Self-important tone
- Bumbling but wise demeanor

**Low learning rate** ensures:
- Base knowledge preserved
- Gentle personality injection
- No catastrophic forgetting
- Smooth integration of character traits

**5% Dogberry mix** balances:
- Enough repetition to learn patterns
- Not so much that Shakespeare is lost
- Natural integration of personality
- Maintains quality while adding character

## Next Steps After Deployment

1. **Collect real-world samples** - Save bot outputs to analyze quality
2. **Iterate on corpus** - Refine synthetic Dogberry text based on outputs
3. **Experiment with percentages** - Try 3%, 7%, 10% Dogberry mix
4. **Try different seeds** - Test various conversation starters
5. **Monitor coherence** - Ensure multi-turn conversations stay in character

---

**Status**: Base training in progress (see `ps aux | grep train_shakespeare`)
**Next**: Generate synthetic Dogberry corpus using guide
