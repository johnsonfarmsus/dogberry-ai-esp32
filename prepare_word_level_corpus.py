#!/usr/bin/env python3
"""
Prepare word-level training corpus for Dogberry bot
- Extract Shakespeare comedies
- Clean stage directions, character names
- Mix with Dogberry dialogue (oversampled to 20-25%)
"""

import os
import re

# Known Shakespeare comedies
COMEDY_FILES = [
    "shakespeare_1103_a_midsummer_night's_dream.txt",
    "shakespeare_1110_as_you_like_it.txt",
    "shakespeare_1111_much_ado_about_nothing.txt",
    "shakespeare_1112_twelfth_night.txt",
    "shakespeare_1113_the_merchant_of_venice.txt",
    "shakespeare_1119_the_taming_of_the_shrew.txt",
    "shakespeare_1120_the_comedy_of_errors.txt",
    "shakespeare_1121_all's_well_that_ends_well.txt",
    "shakespeare_1122_measure_for_measure.txt",
    "shakespeare_1123_the_winter's_tale.txt",
    "shakespeare_1126_the_two_gentlemen_of_verona.txt",
    "shakespeare_1127_love's_labour's_lost.txt",
]

def clean_shakespeare_text(text):
    """Clean Shakespeare text - remove headers, stage directions, etc."""
    lines = text.split('\n')
    cleaned_lines = []
    in_play = False

    for line in lines:
        # Skip Project Gutenberg headers (first ~70 lines usually)
        if 'Actus' in line or 'ACT' in line or 'SCENE' in line:
            in_play = True
            continue

        if not in_play:
            continue

        # Skip stage directions (lines starting with capital Enter, Exit, Exeunt, etc.)
        if re.match(r'^\s*(Enter|Exit|Exeunt|Alarum|Flourish|Sennet|Within)', line, re.IGNORECASE):
            continue

        # Skip character name labels (e.g., "DOGBERRY." or "Dogb.")
        if re.match(r'^\s*[A-Z][a-z]*\.', line):
            # This is a character speaking, remove the name but keep the dialogue
            line = re.sub(r'^\s*[A-Z][a-z]*\.\s*', '', line)

        # Skip lines that are ALL CAPS (usually character names or stage directions)
        if line.isupper() and len(line.strip()) > 0:
            continue

        # Skip empty lines
        if not line.strip():
            continue

        # Remove bracket annotations [like this]
        line = re.sub(r'\[.*?\]', '', line)

        cleaned_lines.append(line.strip())

    return ' '.join(cleaned_lines)

def load_shakespeare_comedies():
    """Load and clean all Shakespeare comedies"""
    print("=" * 70)
    print("LOADING SHAKESPEARE COMEDIES")
    print("=" * 70)

    all_text = []

    for filename in COMEDY_FILES:
        filepath = os.path.join('shakespeare', filename)
        if not os.path.exists(filepath):
            print(f"  ⚠ Warning: {filename} not found, skipping")
            continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        cleaned = clean_shakespeare_text(text)
        all_text.append(cleaned)

        print(f"  ✓ {filename}: {len(cleaned):,} characters")

    combined = ' '.join(all_text)
    print(f"\nTotal Shakespeare: {len(combined):,} characters")

    return combined

def load_dogberry_corpus():
    """Load Dogberry-specific dialogue"""
    print("\n" + "=" * 70)
    print("LOADING DOGBERRY CORPUS")
    print("=" * 70)

    with open('dogberry_corpus_combined.txt', 'r', encoding='utf-8') as f:
        dogberry_text = f.read()

    print(f"  Dogberry corpus: {len(dogberry_text):,} characters")

    return dogberry_text

def mix_corpora(shakespeare_text, dogberry_text, dogberry_target_percent=0.23):
    """
    Mix Shakespeare and Dogberry corpora
    Oversample Dogberry to reach target percentage
    """
    print("\n" + "=" * 70)
    print("MIXING CORPORA")
    print("=" * 70)

    shakespeare_len = len(shakespeare_text)
    dogberry_len = len(dogberry_text)

    # Calculate how many times to repeat Dogberry to hit target percentage
    # Target: dogberry_repeated / (shakespeare + dogberry_repeated) = target_percent
    # Solve: dogberry_repeated = shakespeare * target_percent / (1 - target_percent)

    target_dogberry_len = int(shakespeare_len * dogberry_target_percent / (1 - dogberry_target_percent))
    repeat_factor = target_dogberry_len / dogberry_len

    print(f"  Shakespeare: {shakespeare_len:,} characters")
    print(f"  Dogberry original: {dogberry_len:,} characters")
    print(f"  Target Dogberry percent: {dogberry_target_percent*100:.1f}%")
    print(f"  Repeat factor: {repeat_factor:.1f}x")

    # Repeat Dogberry corpus (use ceil to ensure we hit target)
    import math
    dogberry_repeated = (dogberry_text + ' ') * math.ceil(repeat_factor)

    # Mix them
    combined = shakespeare_text + ' ' + dogberry_repeated

    actual_percent = len(dogberry_repeated) / len(combined)

    print(f"\nFinal corpus:")
    print(f"  Total: {len(combined):,} characters (~{len(combined.split()):,} words)")
    print(f"  Dogberry: {actual_percent*100:.1f}%")

    return combined

def save_corpus(text, filename='corpus_word_level.txt'):
    """Save prepared corpus"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\n✓ Saved to {filename}")
    print(f"  Size: {len(text):,} characters")
    print(f"  Words: ~{len(text.split()):,}")

if __name__ == '__main__':
    # Load Shakespeare comedies
    shakespeare = load_shakespeare_comedies()

    # Load Dogberry corpus
    dogberry = load_dogberry_corpus()

    # Mix them (target 25% Dogberry)
    combined = mix_corpora(shakespeare, dogberry, dogberry_target_percent=0.25)

    # Save
    save_corpus(combined, 'corpus_word_level.txt')

    print("\n" + "=" * 70)
    print("✓ CORPUS PREPARATION COMPLETE!")
    print("=" * 70)
