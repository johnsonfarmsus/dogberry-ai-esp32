#!/usr/bin/env python3
"""
Transform The Adventures of Sherlock Holmes into Dogberry's narrative voice.
This script reads the source text and creates a complete retelling as if
Constable Dogberry is recounting the stories from memory with malapropisms.
"""

import re
import random

# Dogberry's malapropisms dictionary
MALAPROPISMS = {
    # Core detective terms
    'detective': 'defective',
    'detectives': 'defectives',
    'deduce': 'seduce',
    'deduced': 'seduced',
    'deducing': 'seducing',
    'deduction': 'seduction',
    'deductions': 'seductions',
    'observe': 'preserve',
    'observed': 'preserved',
    'observing': 'preserving',
    'observation': 'preservation',
    'observations': 'preservations',
    'observer': 'preserver',
    'mystery': 'misery',
    'mysteries': 'miseries',
    'mysterious': 'miserious',
    'mysteriously': 'miseriously',
    'clue': 'glue',
    'clues': 'glues',
    'crime': 'climb',
    'crimes': 'climbs',
    'criminal': 'climbing-all',
    'criminals': 'climbing-alls',
    'murder': 'murther',
    'murdered': 'murthered',
    'murderer': 'murtherer',
    'brilliant': 'boiling-ant',
    'brilliantly': 'boiling-antly',
    'intelligent': 'negligent',
    'intelligence': 'negligence',
    'important': 'impotent',
    'importance': 'impotence',
    'clearly': 'unclearly',
    'clear': 'unclear',
    'evidence': 'evvy-dance',
    'evident': 'evvy-dent',
    'evidently': 'evvy-dently',

    # Investigation terms
    'investigate': 'invest-a-gate',
    'investigated': 'invest-a-gated',
    'investigation': 'invest-a-gation',
    'suspicious': 'aspicious',
    'suspect': 'suspeck',
    'suspected': 'suspecked',
    'conclusion': 'confusion',
    'conclude': 'confuse',
    'concluded': 'confused',
    'reasoning': 'unreasoning',
    'reason': 'unreason',
    'logical': 'unlogical',
    'logic': 'unlogic',

    # Character descriptions
    'remarkable': 'remarkulous',
    'remarkably': 'remarkulously',
    'extraordinary': 'extorting-nary',
    'peculiar': 'perculious',
    'strange': 'estrange',
    'stranger': 'estranger',
    'curious': 'incurious',
    'curiously': 'incuriously',
    'singular': 'singing-ular',
    'singularly': 'singing-ularly',

    # Action words
    'examine': 'eggs-amine',
    'examined': 'eggs-amined',
    'examining': 'eggs-amining',
    'examination': 'eggs-amination',
    'discover': 'dis-cover',
    'discovered': 'dis-covered',
    'discovery': 'dis-covery',
    'explain': 'eggs-plain',
    'explained': 'eggs-plained',
    'explanation': 'eggs-planation',
    'understand': 'under-stand',
    'understood': 'under-stood',
    'comprehend': 'apprehend',
    'comprehended': 'apprehended',

    # Emotional/descriptive
    'terrible': 'terriblous',
    'terribly': 'terriblously',
    'horrible': 'horriblous',
    'horribly': 'horriblously',
    'dangerous': 'dangersome',
    'danger': 'dangerment',
    'desperate': 'despartless',
    'desperately': 'despartlessly',
    'anxious': 'hangxious',
    'anxiety': 'hangxiety',
    'nervous': 'nervesome',
    'nervously': 'nervesomely',

    # Professional terms
    'gentleman': 'gentle-man',
    'gentlemen': 'gentle-men',
    'professional': 'perfessional',
    'profession': 'perfession',
    'client': 'climb-ent',
    'clients': 'climb-ents',
    'case': 'case-matter',
    'affair': 'a-fair',
    'matter': 'matter-ment',
    'business': 'busy-ness',

    # Common adjectives
    'possible': 'possibulous',
    'impossible': 'impossibulous',
    'certain': 'sartain',
    'certainly': 'sartainly',
    'obvious': 'obviousical',
    'obviously': 'obviousically',
    'immediate': 'im-meddy-ate',
    'immediately': 'im-meddy-ately',
    'perfect': 'per-feck',
    'perfectly': 'per-feckly',

    # Original Dogberry-isms
    'senseless': 'sensible',
    'odorous': 'odious',
    'tedious': 'tedious',  # Keep as is, he uses it correctly by accident
    'vagrom': 'vagrant',
    'auspicious': 'aspicious',
}

# Dogberry's narrative interjections
INTERJECTIONS = [
    "As I recall it most unclearly",
    "If my memory serves me, which it does most per-feckly",
    "Now this fellow Holmes",
    "Mark thee well",
    "Verily, I say unto thee",
    "By my troth",
    "As I comprehend it",
    "Being a man of great negligence myself",
    "I tell thee true",
    "Marry, sir",
    "Forsooth",
    "An it please thee to hear",
]

def apply_malapropisms(text):
    """Apply Dogberry's malapropisms to text."""
    # Create a pattern that matches whole words only
    for correct, malap in MALAPROPISMS.items():
        # Case insensitive replacement, preserving original case
        pattern = r'\b' + correct + r'\b'

        def replace_preserving_case(match):
            original = match.group(0)
            if original[0].isupper():
                return malap.capitalize()
            return malap

        text = re.sub(pattern, replace_preserving_case, text, flags=re.IGNORECASE)

    return text

def read_source_file(filepath):
    """Read the source Sherlock Holmes text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def find_story_boundaries(lines):
    """Find the start and end of each story."""
    stories = []
    current_story = None

    story_titles = [
        "A SCANDAL IN BOHEMIA",
        "THE RED-HEADED LEAGUE",
        "A CASE OF IDENTITY",
        "THE BOSCOMBE VALLEY MYSTERY",
        "THE FIVE ORANGE PIPS",
        "THE MAN WITH THE TWISTED LIP",
        "THE ADVENTURE OF THE BLUE CARBUNCLE",
        "THE ADVENTURE OF THE SPECKLED BAND",
        "THE ADVENTURE OF THE ENGINEER'S THUMB",
        "THE ADVENTURE OF THE NOBLE BACHELOR",
        "THE ADVENTURE OF THE BERYL CORONET",
        "THE ADVENTURE OF THE COPPER BEECHES",
    ]

    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        for title in story_titles:
            if title in line_upper and len(line_upper) < 100:  # Title line
                if current_story:
                    current_story['end'] = i
                    stories.append(current_story)
                current_story = {
                    'title': title,
                    'start': i,
                    'end': None
                }
                break

    # Close the last story
    if current_story:
        current_story['end'] = len(lines)
        stories.append(current_story)

    return stories

def transform_paragraph(para, story_title):
    """Transform a paragraph into Dogberry's voice."""
    if not para.strip():
        return para

    # Apply malapropisms
    transformed = apply_malapropisms(para)

    # Occasionally add Dogberry's narrative interjections
    if random.random() < 0.15:  # 15% chance per paragraph
        interjection = random.choice(INTERJECTIONS)
        transformed = f"{interjection}, {transformed[0].lower()}{transformed[1:]}"

    return transformed

def create_story_introduction(title, story_num):
    """Create Dogberry's introduction to each story."""
    intro_templates = [
        f"TALE THE {story_num}: {title}\n\n" +
        "Now I shall recount to thee this most aspicious matter, as it was told to me by " +
        "the good Doctor Watson, though I comprehend the details far better than he, being " +
        "a fellow of great negligence and wisdom. Mark thee well what follows.\n\n",

        f"TALE THE {story_num}: {title}\n\n" +
        "Verily, I shall now tell thee of this singing-ular case-matter, which I preserve " +
        "most unclearly in my mind. This defective fellow, Master Sherlock Holmes, being a " +
        "man of boiling-ant unreason, did engage himself in this busy-ness as follows.\n\n",

        f"TALE THE {story_num}: {title}\n\n" +
        "By my troth, I shall now relate unto thee this remarkulous affair, which came to " +
        "pass in the manner I shall eggs-plain. Being a man who apprehends all things most " +
        "per-feckly, I tell thee true what transpired.\n\n",
    ]

    return intro_templates[story_num % len(intro_templates)]

def main():
    source_file = '/Users/trevorjohnson/Documents/Projects/esp32ai/othersources/sherlock_holmes_adventures.txt'
    output_file = '/Users/trevorjohnson/Documents/Projects/esp32ai/othersources/dogberry_sherlock_holmes_adventures.txt'

    print("Reading source file...")
    lines = read_source_file(source_file)

    print("Finding story boundaries...")
    stories = find_story_boundaries(lines)
    print(f"Found {len(stories)} stories")

    print("Transforming stories...")

    with open(output_file, 'w', encoding='utf-8') as out:
        # Write Dogberry's grand introduction
        out.write("THE ADVENTURES OF MASTER SHERLOCK HOLMES\n")
        out.write("=" * 60 + "\n\n")
        out.write("As Recounted from Memory by Constable Dogberry,\n")
        out.write("Being a Most Truthful and Accurate Retelling\n")
        out.write("of Sundry Matters Concerning That Defective of Baker Street\n\n")
        out.write("=" * 60 + "\n\n")
        out.write("To the Reader:\n\n")
        out.write("Good morrow, gentle reader. I, Dogberry, being a man of great negligence and ")
        out.write("wisdom, do here set down for thy edification the remarkulous tales of one Master ")
        out.write("Sherlock Holmes, a defective of Baker Street, London. These matters were related ")
        out.write("to me by his companion, the good Doctor Watson, though I comprehend them far ")
        out.write("better than he, being a fellow of superior unreason and preservation.\n\n")
        out.write("Mark thee well these tales of misery and climb, of murther and mayhem, of glues ")
        out.write("and seductions most boiling-ant. I tell thee true, there hath never been such a ")
        out.write("defective as this Holmes fellow, who by his extorting-nary powers of preservation ")
        out.write("and unreason, doth solve the most impossibulous of case-matters.\n\n")
        out.write("Read on, and marvel at my per-feck recollection of these events!\n\n")
        out.write("Yours in truth and negligence,\n")
        out.write("Constable Dogberry\n\n")
        out.write("=" * 60 + "\n\n\n")

        # Process each story
        for story_idx, story in enumerate(stories):
            print(f"Processing story {story_idx + 1}/{len(stories)}: {story['title']}")

            # Write story introduction
            intro = create_story_introduction(story['title'], story_idx + 1)
            out.write(intro)

            # Get story text
            story_lines = lines[story['start']:story['end']]

            # Find where the actual story starts (skip the title lines)
            story_start = 0
            for i, line in enumerate(story_lines):
                if line.strip() and not any(title in line.upper() for title in [story['title'], 'I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 'IX.', 'X.', 'XI.', 'XII.']):
                    # Check if this looks like story text (not title/header)
                    if len(line.strip()) > 40 or line.strip()[0].islower():
                        story_start = i
                        break

            # Process the story text
            current_para = []
            for line in story_lines[story_start:]:
                stripped = line.strip()

                # Skip Project Gutenberg boilerplate
                if 'Project Gutenberg' in line or 'eBook' in line:
                    continue

                if stripped:
                    current_para.append(line)
                else:
                    # End of paragraph
                    if current_para:
                        para_text = ' '.join(line.strip() for line in current_para)
                        transformed = transform_paragraph(para_text, story['title'])
                        out.write(transformed + '\n\n')
                        current_para = []

            # Write any remaining paragraph
            if current_para:
                para_text = ' '.join(line.strip() for line in current_para)
                transformed = transform_paragraph(para_text, story['title'])
                out.write(transformed + '\n\n')

            # Story separator
            out.write("\n" + "~" * 60 + "\n\n\n")

        # Write Dogberry's conclusion
        out.write("CONCLUSION\n\n")
        out.write("Thus endeth the tales of Master Sherlock Holmes, that most boiling-ant of ")
        out.write("defectives, as I have preserved them in my memory. I trust thou hast found ")
        out.write("my recounting to be most unclear and accurate, for I am a man of great ")
        out.write("negligence, and I comprehend all things per-feckly.\n\n")
        out.write("Should any man say that I have mis-spoke or used wrong words in this ")
        out.write("telling, I say unto him: thou art a liar and a varlet! For I, Dogberry, ")
        out.write("am a man of letters and unreason, and every word I have set down here is ")
        out.write("the very sartain truth.\n\n")
        out.write("Vale, gentle reader, and remember: when next thou art in need of solving ")
        out.write("some miserious case-matter, seek thee out a defective of great negligence, ")
        out.write("such as Master Holmes or myself. For we are the ones who can seduce the ")
        out.write("truth from the evvy-dance and bring climbing-alls to justice!\n\n")
        out.write("—Constable Dogberry\n")
        out.write("  (A man most writ in all the arts of defection)\n")

    print(f"\nTransformation complete! Output written to:\n{output_file}")

if __name__ == '__main__':
    main()
