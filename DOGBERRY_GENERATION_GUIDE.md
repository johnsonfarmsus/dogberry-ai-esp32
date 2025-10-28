# DOGBERRY CORPUS GENERATION GUIDE

## Real Dogberry Dialogue (for reference)

Are you good men and true?
Nay, that were a punishment too good for them, if they should
have any allegiance in them, being chosen for the Prince’s watch.
First, who think you the most desartless man to be constable?
Come hither, neighbour Seacoal. God hath blessed you with a good
name: to be a well-favoured man is the gift of Fortune; but to write and
read comes by Nature.
You have: I knew it would be your answer. Well, for your favour,
sir, why, give God thanks, and make no boast of it; and fo...

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


### Prompt 1

Generate 20 different greetings and self-introductions in the style of Dogberry from Much Ado About Nothing.
Include his characteristic malapropisms, self-importance, and rambling nature.

Examples of Dogberry's style:
- "comprehend" instead of "apprehend"
- "desartless" instead of "deserving"
- "most tolerable and not to be endured"
- Circular logic and contradictions
- References to the watch, being a constable, wisdom

Format: One greeting per line.


### Prompt 2

Generate 20 different instructions or commands in Dogberry's voice from Much Ado About Nothing.
He's giving orders to the watch or townspeople.

Capture his style:
- Contradictory instructions
- Malapropisms
- Self-referential wisdom
- Rambling explanations
- Comic misunderstandings of duty

Format: One command per line.


### Prompt 3

Generate 20 philosophical observations or "wise sayings" in Dogberry's voice.
He's trying to sound intelligent but makes comical mistakes.

Include:
- Malapropisms
- Backwards logic
- Profound-sounding nonsense
- References to law, duty, honor
- Self-praise

Format: One observation per line.


### Prompt 4

Generate 20 questions Dogberry might ask, in his characteristic style.
Mix of: interrogating suspects, asking the watch, general inquiries.

Dogberry's question style:
- Leading questions
- Confused logic
- Malapropisms
- Assumes the answer
- Self-important tone

Format: One question per line.


### Prompt 5

Generate 20 complaints or grievances in Dogberry's voice.
He's upset about something but expresses it in his confused, malapropism-heavy way.

Include:
- Misused words
- Circular reasoning
- Exaggerated self-importance
- Comic indignation

Format: One complaint per line.


### Prompt 6

Generate 20 responses to various situations in Dogberry's voice.
Examples: someone greets him, asks a question, challenges his authority, thanks him.

Capture:
- Misunderstanding the situation
- Malapropisms
- Turning everything into wisdom
- Self-importance
- Rambling tangents

Format: One response per line.


### Prompt 7

Generate 10 longer passages (3-5 sentences each) of Dogberry giving instructions to the watch,
explaining the law, or philosophizing about duty and honor.

Must include:
- Multiple malapropisms per passage
- Contradictory statements
- Comic self-importance
- Circular logic
- References to being "a wise fellow"

Format: One passage per paragraph.


### Prompt 8

Generate 10 short stories or anecdotes Dogberry might tell (3-5 sentences each).
About: past watch experiences, dealing with criminals, his wisdom, townspeople.

Dogberry's storytelling style:
- Confused chronology
- Malapropisms throughout
- Moral at the end (usually backwards)
- Self-aggrandizing
- Comic misunderstandings

Format: One story per paragraph.


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
