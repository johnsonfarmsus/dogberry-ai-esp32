"""
Generate Synthetic Dogberry Training Corpus

Uses Claude API to generate authentic Dogberry-style dialogue
for fine-tuning the Shakespeare model with personality.

Strategy:
1. Extract real Dogberry dialogue as examples
2. Generate 50-100KB of synthetic Dogberry text
3. Mix with original Shakespeare for fine-tuning
"""

import re
from pathlib import Path

def extract_real_dogberry_dialogue():
    """Extract all actual Dogberry lines from Shakespeare"""
    print("Extracting real Dogberry dialogue...")

    with open('shakespeare/shakespeare_100_complete_works_of_william_shakespeare.txt', 'r') as f:
        text = f.read()

    lines = []
    is_dogberry = False

    for line in text.split('\n'):
        line = line.strip()

        if line and line.isupper() and '.' in line:
            if 'DOGBERRY' in line:
                is_dogberry = True
            else:
                is_dogberry = False
        elif is_dogberry and line:
            if not line.startswith('[') and not line.endswith(']'):
                lines.append(line)

    dialogue = '\n'.join(lines)
    print(f"  Found {len(lines)} lines ({len(dialogue)} characters)")

    # Save for reference
    with open('dogberry_real_dialogue.txt', 'w') as f:
        f.write(dialogue)

    return dialogue


def create_generation_prompts():
    """Create prompts for generating Dogberry-style text"""

    prompts = [
        # Greetings and introductions
        """Generate 20 different greetings and self-introductions in the style of Dogberry from Much Ado About Nothing.
Include his characteristic malapropisms, self-importance, and rambling nature.

Examples of Dogberry's style:
- "comprehend" instead of "apprehend"
- "desartless" instead of "deserving"
- "most tolerable and not to be endured"
- Circular logic and contradictions
- References to the watch, being a constable, wisdom

Format: One greeting per line.""",

        # Instructions and commands
        """Generate 20 different instructions or commands in Dogberry's voice from Much Ado About Nothing.
He's giving orders to the watch or townspeople.

Capture his style:
- Contradictory instructions
- Malapropisms
- Self-referential wisdom
- Rambling explanations
- Comic misunderstandings of duty

Format: One command per line.""",

        # Observations and wisdom
        """Generate 20 philosophical observations or "wise sayings" in Dogberry's voice.
He's trying to sound intelligent but makes comical mistakes.

Include:
- Malapropisms
- Backwards logic
- Profound-sounding nonsense
- References to law, duty, honor
- Self-praise

Format: One observation per line.""",

        # Questions
        """Generate 20 questions Dogberry might ask, in his characteristic style.
Mix of: interrogating suspects, asking the watch, general inquiries.

Dogberry's question style:
- Leading questions
- Confused logic
- Malapropisms
- Assumes the answer
- Self-important tone

Format: One question per line.""",

        # Complaints and grievances
        """Generate 20 complaints or grievances in Dogberry's voice.
He's upset about something but expresses it in his confused, malapropism-heavy way.

Include:
- Misused words
- Circular reasoning
- Exaggerated self-importance
- Comic indignation

Format: One complaint per line.""",

        # Responses to situations
        """Generate 20 responses to various situations in Dogberry's voice.
Examples: someone greets him, asks a question, challenges his authority, thanks him.

Capture:
- Misunderstanding the situation
- Malapropisms
- Turning everything into wisdom
- Self-importance
- Rambling tangents

Format: One response per line.""",

        # Long-form monologues
        """Generate 10 longer passages (3-5 sentences each) of Dogberry giving instructions to the watch,
explaining the law, or philosophizing about duty and honor.

Must include:
- Multiple malapropisms per passage
- Contradictory statements
- Comic self-importance
- Circular logic
- References to being "a wise fellow"

Format: One passage per paragraph.""",

        # Storytelling
        """Generate 10 short stories or anecdotes Dogberry might tell (3-5 sentences each).
About: past watch experiences, dealing with criminals, his wisdom, townspeople.

Dogberry's storytelling style:
- Confused chronology
- Malapropisms throughout
- Moral at the end (usually backwards)
- Self-aggrandizing
- Comic misunderstandings

Format: One story per paragraph.""",
    ]

    return prompts


def save_generation_guide():
    """Save a guide for manually generating Dogberry text"""

    real_dialogue = extract_real_dogberry_dialogue()
    prompts = create_generation_prompts()

    guide = f"""# DOGBERRY CORPUS GENERATION GUIDE

## Real Dogberry Dialogue (for reference)

{real_dialogue[:500]}...

[Full dialogue saved to: dogberry_real_dialogue.txt]

## Generation Strategy

Use Claude or GPT-4 to generate ~50-100KB of Dogberry-style text.

**Key Characteristics:**
1. **Malapropisms** - wrong words that sound similar
   - "comprehend" (apprehend)
   - "desartless" (deserving)
   - "vagrom" (vagrant)
   - "most tolerable" (intolerable)
   - "excommunication" (examination)

2. **Contradictory Logic**
   - "most tolerable and not to be endured"
   - "if they will not, then let them"

3. **Self-Important Rambling**
   - References to being "a wise fellow"
   - Circular explanations
   - Over-complicating simple ideas

4. **Comic Incompetence**
   - Misunderstanding his duties
   - Backwards priorities
   - Missing the obvious

## Generation Prompts

Use these prompts with Claude/GPT-4 to generate text:

"""

    for i, prompt in enumerate(prompts, 1):
        guide += f"\n### Prompt {i}\n\n{prompt}\n\n"

    guide += """
## How to Use

1. **Generate Text:**
   - Use each prompt with Claude or GPT-4
   - Generate multiple batches for variety
   - Aim for 10-20KB per prompt (~50-100KB total)

2. **Save Output:**
   - Save all generated text to: `dogberry_synthetic_corpus.txt`
   - One line per dialogue/response
   - Keep paragraphs for longer pieces

3. **Quality Check:**
   - Read through for authenticity
   - Remove any modern references
   - Ensure malapropisms are present
   - Check for Dogberry's voice

4. **Prepare for Training:**
   - Run `prepare_finetuning_data.py`
   - This will mix synthetic Dogberry with Shakespeare
   - Create training corpus with proper balance

## Target Corpus Size

- **Real Dogberry:** ~10KB (included)
- **Synthetic Dogberry:** 50-100KB (to generate)
- **Original Shakespeare:** 5.3MB (existing)

**Fine-tuning mix:**
- 95% Original Shakespeare
- 5% Dogberry (real + synthetic)

This gives the model enough exposure to learn patterns
without overwhelming the Shakespeare quality.

## Example Generated Text

**Good examples:**

"Good morrow, friend! I am a constable, which is to say, a most desartless
officer of the watch. We shall comprehend all vagrom men tonight, for 'tis
our duty to be both vigilant and most peaceable in our apprehensions."

"What say you to the watch? For we are men of good standing, which is to
say, we stand upon our honor and our feet, being neither cowards nor
excommunicants in the eyes of the law."

**Bad examples (avoid):**

"Hey there! I'm Dogberry." ❌ (too modern)
"I correctly apprehend the situation." ❌ (no malapropisms)
"Let me be clear and concise." ❌ (not rambling enough)

## Next Steps

After generating corpus:
1. Save to `dogberry_synthetic_corpus.txt`
2. Run `prepare_finetuning_data.py`
3. Run `finetune_dogberry.py`
4. Export with `export_tflite.py`
5. Deploy to ESP32!
"""

    with open('DOGBERRY_GENERATION_GUIDE.md', 'w') as f:
        f.write(guide)

    print("\n✓ Guide created: DOGBERRY_GENERATION_GUIDE.md")
    print("\nNext steps:")
    print("1. Use Claude/GPT-4 to generate text following the prompts")
    print("2. Save to: dogberry_synthetic_corpus.txt")
    print("3. Run: prepare_finetuning_data.py")
    print("4. Run: finetune_dogberry.py")


if __name__ == '__main__':
    print("=" * 70)
    print("DOGBERRY CORPUS GENERATION SETUP")
    print("=" * 70)
    save_generation_guide()
