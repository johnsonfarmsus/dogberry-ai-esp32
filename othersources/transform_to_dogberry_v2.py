#!/usr/bin/env python3
"""
Transform The Adventures of Sherlock Holmes into Dogberry's narrative voice.
This creates a complete retelling where Dogberry is actively NARRATING the stories,
not just replacing words.
"""

import re

# Comprehensive malapropism dictionary
MALAPROPISMS = {
    # Detective/investigation terms
    r'\bdetective(s?)\b': r'defective\1',
    r'\bDeduc(e|ed|ing|tion|tions)\b': r'Seduc\1',
    r'\bdeduc(e|ed|ing|tion|tions)\b': r'seduc\1',
    r'\bObserv(e|ed|ing|ation|ations|er)\b': r'Preserv\1',
    r'\bobserv(e|ed|ing|ation|ations|er)\b': r'preserv\1',
    r'\bMyster(y|ies|ious|iously)\b': r'Miser\1',
    r'\bmyster(y|ies|ious|iously)\b': r'miser\1',
    r'\bclue(s?)\b': r'glue\1',
    r'\bCrime(s?)\b': r'Climb\1',
    r'\bcrime(s?)\b': r'climb\1',
    r'\bCriminal(s?)\b': r'Climbing-all\1',
    r'\bcriminal(s?)\b': r'climbing-all\1',
    r'\bMurder(s?|ed|er|ers|ous)\b': r'Murther\1',
    r'\bmurder(s?|ed|er|ers|ous)\b': r'murther\1',

    # Intelligence/reasoning terms
    r'\bBrilliant(ly)?\b': r'Boiling-ant\1',
    r'\bbrilliant(ly)?\b': r'boiling-ant\1',
    r'\bIntelligen(t|ce)\b': r'Negligen\1',
    r'\bintelligen(t|ce)\b': r'negligen\1',
    r'\bReason(ing|ed|er)?\b': r'Unreason\1',
    r'\breason(ing|ed|er)?\b': r'unreason\1',
    r'\bLogic(al)?\b': r'Unlogic\1',
    r'\blogic(al)?\b': r'unlogic\1',

    # Importance/clarity terms
    r'\bImportan(t|ce)\b': r'Impoten\1',
    r'\bimportan(t|ce)\b': r'impoten\1',
    r'\bClear(ly)?\b': r'Unclear\1',
    r'\bclear(ly)?\b': r'unclear\1',
    r'\bEviden(t|ce|tly)\b': r'Evvy-den\1',
    r'\beviden(t|ce|tly)\b': r'evvy-den\1',
    r'\bObvious(ly)?\b': r'Obviousical\1',
    r'\bobvious(ly)?\b': r'obviousical\1',

    # Investigation verbs
    r'\bInvestigat(e|ed|ing|ion)\b': r'Invest-a-gat\1',
    r'\binvestigat(e|ed|ing|ion)\b': r'invest-a-gat\1',
    r'\bExamin(e|ed|ing|ation)\b': r'Eggs-amin\1',
    r'\bexamin(e|ed|ing|ation)\b': r'eggs-amin\1',
    r'\bDiscover(ed|y|ies)?\b': r'Dis-cover\1',
    r'\bdiscover(ed|y|ies)?\b': r'dis-cover\1',
    r'\bExplain(ed|ing)?\b': r'Eggs-plain\1',
    r'\bexplain(ed|ing)?\b': r'eggs-plain\1',
    r'\bExplanation\b': r'Eggs-planation',
    r'\bexplanation\b': r'eggs-planation',

    # Suspicion/conclusion terms
    r'\bSuspicious\b': r'Aspicious',
    r'\bsuspicious\b': r'aspicious',
    r'\bSuspect(ed)?\b': r'Suspeck\1',
    r'\bsuspect(ed)?\b': r'suspeck\1',
    r'\bConclu(de|ded|sion)\b': r'Confu\1',
    r'\bconclu(de|ded|sion)\b': r'confu\1',
    r'\bComprehend(ed)?\b': r'Apprehend\1',
    r'\bcomprehend(ed)?\b': r'apprehend\1',

    # Descriptive adjectives
    r'\bRemarkabl(e|y)\b': r'Remarkulous\1',
    r'\bremarkabl(e|y)\b': r'remarkulous\1',
    r'\bExtraordinary\b': r'Extorting-nary',
    r'\bextraordinary\b': r'extorting-nary',
    r'\bPeculiar\b': r'Perculious',
    r'\bpeculiar\b': r'perculious',
    r'\bStrange(r?)\b': r'Estrange\1',
    r'\bstrange(r?)\b': r'estrange\1',
    r'\bCurious(ly)?\b': r'Incurious\1',
    r'\bcurious(ly)?\b': r'incurious\1',
    r'\bSingular(ly)?\b': r'Singing-ular\1',
    r'\bsingular(ly)?\b': r'singing-ular\1',

    # Emotions/states
    r'\bTerribl(e|y)\b': r'Terriblous\1',
    r'\bterribl(e|y)\b': r'terriblous\1',
    r'\bHorribl(e|y)\b': r'Horriblous\1',
    r'\bhorribl(e|y)\b': r'horriblous\1',
    r'\bDangerous\b': r'Dangersome',
    r'\bdangerous\b': r'dangersome',
    r'\bDanger\b': r'Dangerment',
    r'\bdanger\b': r'dangerment',
    r'\bDesperat(e|ely)\b': r'Despartless\1',
    r'\bdesperat(e|ely)\b': r'despartless\1',
    r'\bAnxious\b': r'Hangxious',
    r'\banxious\b': r'hangxious',
    r'\bAnxiety\b': r'Hangxiety',
    r'\banxiety\b': r'hangxiety',
    r'\bNervous(ly)?\b': r'Nervesome\1',
    r'\bnervous(ly)?\b': r'nervesome\1',

    # Professional terms
    r'\bGentleman\b': r'Gentle-man',
    r'\bgentleman\b': r'gentle-man',
    r'\bGentlemen\b': r'Gentle-men',
    r'\bgentlemen\b': r'gentle-men',
    r'\bProfession(al)?\b': r'Perfession\1',
    r'\bprofession(al)?\b': r'perfession\1',
    r'\bClient(s?)\b': r'Climb-ent\1',
    r'\bclient(s?)\b': r'climb-ent\1',
    r'\bCase\b': r'Case-matter',
    r'\bcase\b': r'case-matter',
    r'\bAffair(s?)\b': r'A-fair\1',
    r'\baffair(s?)\b': r'a-fair\1',
    r'\bBusiness\b': r'Busy-ness',
    r'\bbusiness\b': r'busy-ness',

    # Common modifiers
    r'\bPossibl(e|y)\b': r'Possibulous\1',
    r'\bpossibl(e|y)\b': r'possibulous\1',
    r'\bImpossibl(e|y)\b': r'Impossibulous\1',
    r'\bimpossibl(e|y)\b': r'impossibulous\1',
    r'\bCertain(ly)?\b': r'Sartain\1',
    r'\bcertain(ly)?\b': r'sartain\1',
    r'\bImmediate(ly)?\b': r'Im-meddy-ate\1',
    r'\bimmediate(ly)?\b': r'im-meddy-ate\1',
    r'\bPerfect(ly)?\b': r'Per-feck\1',
    r'\bperfect(ly)?\b': r'per-feck\1',
}

def apply_malapropisms(text):
    """Apply Dogberry's malapropisms using regex patterns."""
    for pattern, replacement in MALAPROPISMS.items():
        text = re.sub(pattern, replacement, text)
    return text

def add_dogberry_flourishes(text):
    """Add occasional Dogberry-style narrative flourishes."""
    # Add narrative markers at sentence beginnings occasionally
    sentences = re.split(r'([.!?])\s+', text)
    result = []

    dogberry_intros = [
        "Mark thee well, ",
        "Verily, ",
        "By my troth, ",
        "Forsooth, ",
        "I tell thee true, ",
        "An it please thee to hear, ",
        "As I recall it most unclearly, ",
        "If my memory serves me, ",
        "Being a man of great negligence myself, ",
    ]

    for i, part in enumerate(sentences):
        if i % 20 == 0 and i > 0 and len(part) > 20:  # Every ~10th sentence
            import random
            if random.random() < 0.3:  # 30% chance
                part = dogberry_intros[i % len(dogberry_intros)] + part[0].lower() + part[1:]
        result.append(part)

    return ''.join(result)

def read_source_file(filepath):
    """Read the source text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def find_story_starts(text):
    """Find where each story actually starts (after the title)."""
    # Find the main content start
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    start_pos = text.find(start_marker)
    if start_pos == -1:
        start_pos = 0

    # Find end marker
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    end_pos = text.find(end_marker)
    if end_pos == -1:
        end_pos = len(text)

    return text[start_pos:end_pos]

def main():
    source_file = '/Users/trevorjohnson/Documents/Projects/esp32ai/othersources/sherlock_holmes_adventures.txt'
    output_file = '/Users/trevorjohnson/Documents/Projects/esp32ai/othersources/dogberry_sherlock_holmes_adventures.txt'

    print("Reading source file...")
    source_text = read_source_file(source_file)

    print("Extracting main content...")
    main_content = find_story_starts(source_text)

    print("Applying Dogberry's malapropisms...")
    transformed = apply_malapropisms(main_content)

    print("Adding Dogberry's narrative flourishes...")
    transformed = add_dogberry_flourishes(transformed)

    print("Writing output file...")
    with open(output_file, 'w', encoding='utf-8') as out:
        # Grand introduction
        out.write("=" * 70 + "\n")
        out.write("THE ADVENTURES OF MASTER SHERLOCK HOLMES\n")
        out.write("=" * 70 + "\n\n")
        out.write("As Recounted from Memory by Constable Dogberry\n")
        out.write("Being a Most Truthful and Accurate Retelling of Sundry Matters\n")
        out.write("Concerning That Defective of Baker Street\n\n")
        out.write("=" * 70 + "\n\n")

        out.write("TO THE READER:\n\n")
        out.write("Good morrow, gentle-man reader. I, Dogberry, being a man of great\n")
        out.write("negligence and wisdom, do here set down for thy edification the\n")
        out.write("remarkulous tales of one Master Sherlock Holmes, a defective of\n")
        out.write("Baker Street, London.\n\n")

        out.write("These matter-ments were related to me by his companion, the good\n")
        out.write("Doctor Watson, though I apprehend them far better than he, being\n")
        out.write("a fellow of superior unreason and preservation.\n\n")

        out.write("Mark thee well these tales of misery and climb, of murther and\n")
        out.write("mayhem, of glues and seductions most boiling-ant. I tell thee true,\n")
        out.write("there hath never been such a defective as this Holmes fellow, who\n")
        out.write("by his extorting-nary powers of preservation and unreason, doth\n")
        out.write("solve the most impossibulous of case-matter-ments.\n\n")

        out.write("Read on, and marvel at my per-feck recollection of these events!\n\n")
        out.write("Yours in truth and negligence,\n")
        out.write("Constable Dogberry\n")
        out.write("(A man most writ in all the arts of defection)\n\n")
        out.write("=" * 70 + "\n\n\n")

        # Write the transformed text
        out.write(transformed)

        # Conclusion
        out.write("\n\n" + "=" * 70 + "\n\n")
        out.write("CONCLUSION\n\n")
        out.write("Thus endeth the tales of Master Sherlock Holmes, that most boiling-ant\n")
        out.write("of defectives, as I have preserved them in my memory. I trust thou\n")
        out.write("hast found my recounting to be most unclear and accurate, for I am\n")
        out.write("a man of great negligence, and I apprehend all things per-feckly.\n\n")

        out.write("Should any man say that I have mis-spoke or used wrong words in\n")
        out.write("this telling, I say unto him: thou art a liar and a varlet! For I,\n")
        out.write("Dogberry, am a man of letters and unreason, and every word I have\n")
        out.write("set down here is the very sartain truth.\n\n")

        out.write("Vale, gentle-man reader, and remember: when next thou art in need\n")
        out.write("of solving some miserious case-matter-ment, seek thee out a defective\n")
        out.write("of great negligence, such as Master Holmes or myself. For we are the\n")
        out.write("ones who can seduce the truth from the evvy-dence and bring\n")
        out.write("climbing-alls to justice!\n\n")

        out.write("—Constable Dogberry\n\n")
        out.write("=" * 70 + "\n")

    print(f"\nTransformation complete!")
    print(f"Output written to: {output_file}")

    # Show stats
    import os
    size = os.path.getsize(output_file)
    print(f"Output file size: {size:,} bytes")

if __name__ == '__main__':
    main()
