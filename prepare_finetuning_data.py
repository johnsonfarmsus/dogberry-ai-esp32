"""
Prepare Fine-Tuning Data

Combines real Dogberry dialogue with synthetic corpus
to create the final fine-tuning dataset.

Usage:
    python3 prepare_finetuning_data.py

Requirements:
    - dogberry_real_dialogue.txt (from generate_dogberry_corpus.py)
    - dogberry_synthetic_corpus.txt (manually generated using guide)

Output:
    - dogberry_corpus_combined.txt (ready for fine-tuning)
"""

import os

def load_corpus_file(filepath):
    """Load a corpus file if it exists"""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    return text.strip()

def combine_corpora():
    """Combine real and synthetic Dogberry corpora"""
    print("=" * 70)
    print("DOGBERRY FINE-TUNING DATA PREPARATION")
    print("=" * 70)

    # Load real dialogue
    print("\n1. Loading real Dogberry dialogue...")
    real_dialogue = load_corpus_file('dogberry_real_dialogue.txt')

    if real_dialogue:
        print(f"   ✓ Found: {len(real_dialogue):,} characters")
        print(f"   Lines: {len(real_dialogue.splitlines())}")
    else:
        print("   ⚠️  Not found: dogberry_real_dialogue.txt")
        print("   Run: python3 generate_dogberry_corpus.py")
        real_dialogue = ""

    # Load synthetic corpus
    print("\n2. Loading synthetic Dogberry corpus...")
    synthetic_corpus = load_corpus_file('dogberry_synthetic_corpus.txt')

    if synthetic_corpus:
        print(f"   ✓ Found: {len(synthetic_corpus):,} characters")
        print(f"   Lines: {len(synthetic_corpus.splitlines())}")
    else:
        print("   ⚠️  Not found: dogberry_synthetic_corpus.txt")
        print("   ")
        print("   To create synthetic corpus:")
        print("   - Use Claude/GPT-4 with prompts from DOGBERRY_GENERATION_GUIDE.md")
        print("   - Generate 50-100KB of Dogberry-style dialogue")
        print("   - Save to dogberry_synthetic_corpus.txt")
        print("   ")
        synthetic_corpus = ""

    # Combine
    if not real_dialogue and not synthetic_corpus:
        print("\n❌ ERROR: No corpus data found!")
        print("Cannot proceed with fine-tuning preparation.")
        return False

    print("\n3. Combining corpora...")

    parts = []
    if real_dialogue:
        parts.append(real_dialogue)
    if synthetic_corpus:
        parts.append(synthetic_corpus)

    combined = '\n\n'.join(parts)

    # Save combined corpus
    output_path = 'dogberry_corpus_combined.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"   ✓ Saved: {output_path}")

    # Statistics
    print("\n" + "=" * 70)
    print("CORPUS STATISTICS")
    print("=" * 70)

    print(f"\nReal Dogberry dialogue:")
    print(f"  Characters: {len(real_dialogue):,}")
    print(f"  Lines: {len(real_dialogue.splitlines())}")
    print(f"  Percentage: {len(real_dialogue)/len(combined)*100:.1f}%")

    print(f"\nSynthetic Dogberry corpus:")
    print(f"  Characters: {len(synthetic_corpus):,}")
    print(f"  Lines: {len(synthetic_corpus.splitlines())}")
    print(f"  Percentage: {len(synthetic_corpus)/len(combined)*100:.1f}%")

    print(f"\nCombined corpus:")
    print(f"  Total characters: {len(combined):,}")
    print(f"  Total lines: {len(combined.splitlines())}")

    # Load Shakespeare for comparison
    shakespeare_path = 'shakespeare/shakespeare_100_complete_works_of_william_shakespeare.txt'
    if os.path.exists(shakespeare_path):
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            shakespeare = f.read()

        print(f"\nShakespeare corpus:")
        print(f"  Total characters: {len(shakespeare):,}")

        print(f"\nFine-tuning mix (target ~5% Dogberry):")
        print(f"  Dogberry percentage: {len(combined)/len(shakespeare)*100:.1f}%")

        if len(combined) / len(shakespeare) < 0.03:
            print(f"  ⚠️  Corpus is a bit small - recommend 50-100KB")
        elif len(combined) / len(shakespeare) > 0.10:
            print(f"  ⚠️  Corpus is large - may overwhelm Shakespeare base")
        else:
            print(f"  ✓ Good size for fine-tuning!")

    print("\n" + "=" * 70)
    print("✓ DATA PREPARATION COMPLETE!")
    print("\nNext step:")
    print("  python3 finetune_dogberry.py")
    print("=" * 70)

    return True

def main():
    success = combine_corpora()

    if not success:
        print("\nPlease generate corpus files and try again.")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
