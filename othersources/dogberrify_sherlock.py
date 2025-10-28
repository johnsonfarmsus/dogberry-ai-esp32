#!/usr/bin/env python3
"""
Transform Sherlock Holmes Adventures into Dogberry's narration
"""

import re

# Comprehensive malapropism mappings for Sherlock Holmes
MALAPROPISMS = {
    # Original Dogberry malapropisms
    r'\bapprehend\b': 'comprehend',
    r'\bapprehended\b': 'comprehended',
    r'\bsuspicious\b': 'aspicious',
    r'\bsensible\b': 'senseless',
    r'\bexamination\b': 'excommunication',
    r'\bexamine\b': 'excommunicate',
    r'\bdeserving\b': 'desartless',
    r'\bvagrant\b': 'vagrom',
    r'\bintolerable\b': 'tolerable',
    r'\bmalefactor\b': 'benefactor',
    r'\bodious\b': 'odorous',
    r'\bdamnation\b': 'redemption',

    # Detective/Mystery terms
    r'\bdetective\b': 'defective',
    r'\bdetectives\b': 'defectives',
    r'\bdeduce\b': 'seduce',
    r'\bdeduced\b': 'seduced',
    r'\bdeduction\b': 'seduction',
    r'\bdeductions\b': 'seductions',
    r'\bobserve\b': 'preserve',
    r'\bobserved\b': 'preserved',
    r'\bobservation\b': 'preservation',
    r'\bobservations\b': 'preservations',
    r'\bmystery\b': 'misery',
    r'\bmysteries\b': 'miseries',
    r'\bmysterious\b': 'miser-us',
    r'\bclue\b': 'glue',
    r'\bclues\b': 'glues',
    r'\bevidence\b': 'evvy-dance',
    r'\binvestigate\b': 'in-vest-igate',
    r'\binvestigation\b': 'in-vest-igation',
    r'\binvestigating\b': 'in-vest-igating',
    r'\binvestigator\b': 'in-vest-igator',
    r'\bcrime\b': 'climb',
    r'\bcrimes\b': 'climbs',
    r'\bcriminal\b': 'climb-in-all',
    r'\bcriminals\b': 'climb-in-alls',
    r'\bmurder\b': 'murther',
    r'\bmurderer\b': 'murtherer',
    r'\bmurdered\b': 'murthered',

    # Intelligence/Reasoning terms
    r'\bbrilliant\b': 'boiling-ant',
    r'\bintelligent\b': 'negligent',
    r'\bintelligence\b': 'negligence',
    r'\breason\b': 'raisin',
    r'\breasoning\b': 'raisin-ing',
    r'\breasoned\b': 'raisin-ed',
    r'\blogic\b': 'logic-all',
    r'\blogical\b': 'logic-kill',
    r'\banalysis\b': 'anna-lice-is',
    r'\banalyze\b': 'anna-lies',
    r'\banalyzed\b': 'anna-lied',
    r'\bconclusion\b': 'con-clue-shun',
    r'\bconclude\b': 'con-glued',
    r'\bconcluded\b': 'con-glue-did',

    # Important/Serious terms
    r'\bimportant\b': 'impotent',
    r'\bimportance\b': 'impotence',
    r'\bcrucial\b': 'cruel-shall',
    r'\bcritical\b': 'crit-tickle',
    r'\bserious\b': 'luxurious',
    r'\bseriously\b': 'luxuriously',
    r'\bgrave\b': 'gravy',
    r'\bvital\b': 'vile-tall',
    r'\bessential\b': 'ex-scent-shawl',
    r'\burgent\b': 'her-gent',

    # Clear/Certain terms
    r'\bclear\b': 'unclear',
    r'\bclearly\b': 'unclearly',
    r'\bcertain\b': 'curtain',
    r'\bcertainly\b': 'curtainly',
    r'\bobvious\b': 'ob-vicious',
    r'\bobviously\b': 'ob-viciously',
    r'\bevident\b': 'evvy-dent',
    r'\bevidently\b': 'evvy-dently',
    r'\bapparent\b': 'a-parent',
    r'\bapparently\b': 'a-parently',

    # Explain/Describe terms
    r'\bexplain\b': 'complain',
    r'\bexplained\b': 'complained',
    r'\bexplanation\b': 'complaination',
    r'\bdescribe\b': 'de-scribe',
    r'\bdescribed\b': 'de-scribed',
    r'\bdescription\b': 'de-scription',
    r'\bnarrate\b': 'narrow-ate',
    r'\bnarration\b': 'narrow-ashun',
    r'\brecount\b': 're-count',

    # Strange/Unusual terms
    r'\bstrange\b': 'estranged',
    r'\bstrangely\b': 'estrangedly',
    r'\bpeculiar\b': 'pee-cool-ear',
    r'\bodd\b': 'awed',
    r'\bcurious\b': 'cure-us',
    r'\bunusual\b': 'un-you-shawl',
    r'\bremarkable\b': 'remarkulous',
    r'\bextraordinary\b': 'extorting-nary',

    # Good/Excellent terms
    r'\bexcellent\b': 'excrement',
    r'\bsuperb\b': 'super-burb',
    r'\bsplendid\b': 'spell-ended',
    r'\bmagnificent\b': 'mag-niffy-scent',
    r'\bfine\b': 'fined',
    r'\badmirable\b': 'ad-mire-a-bull',

    # Bad/Terrible terms
    r'\bterrible\b': 'tearable',
    r'\bhorrify\b': 'whore-ify',
    r'\bhorrible\b': 'whore-able',
    r'\bdreadful\b': 'dread-full',
    r'\bawful\b': 'off-full',
    r'\bshocking\b': 'sock-king',

    # Truth/Honest terms
    r'\btruth\b': 'troof',
    r'\btruthful\b': 'troof-full',
    r'\bhonest\b': 'on-ist',
    r'\bhonesty\b': 'on-isty',
    r'\bconfess\b': 'profess',
    r'\bconfessed\b': 'professed',
    r'\bconfession\b': 'profession',
    r'\badmit\b': 'add-mitt',
    r'\badmitted\b': 'add-mitted',

    # Absolute/Complete terms
    r'\babsolute\b': 'obsolete',
    r'\babsolutely\b': 'obsoletely',
    r'\bcomplete\b': 'comp-leet',
    r'\bcompletely\b': 'comp-leetly',
    r'\bentire\b': 'en-tire',
    r'\bentirely\b': 'en-tirely',
    r'\btotal\b': 'toad-all',
    r'\btotally\b': 'toad-ally',

    # Precise/Exact terms
    r'\bprecise\b': 'preciess',
    r'\bprecisely\b': 'preciessly',
    r'\bexact\b': 'ex-act',
    r'\bexactly\b': 'ex-actly',
    r'\baccurate\b': 'ack-your-ate',
    r'\baccurately\b': 'ack-your-ately',

    # Danger/Safe terms
    r'\bdanger\b': 'dane-jer',
    r'\bdangerous\b': 'dane-jer-us',
    r'\bperil\b': 'pair-all',
    r'\bperilous\b': 'pair-all-us',
    r'\bsafe\b': 'saif',
    r'\bsafety\b': 'saif-ty',
    r'\bsecure\b': 'see-cure',
    r'\bsecurity\b': 'see-cure-ity',

    # Person terms
    r'\bgentleman\b': 'gentle-man',
    r'\bgentlemen\b': 'gentle-men',
    r'\blady\b': 'lay-dee',
    r'\bladies\b': 'lay-dees',
    r'\bperson\b': 'purr-son',
    r'\bpersons\b': 'purr-sons',
    r'\bindividual\b': 'in-divvy-duel',
    r'\bcharacter\b': 'care-acter',

    # Adventure/Story terms
    r'\badventure\b': 'advent-chore',
    r'\badventures\b': 'advent-chores',
    r'\baffair\b': 'a-fair',
    r'\baffairs\b': 'a-fairs',
    r'\bepisode\b': 'epp-is-owed',
    r'\bincident\b': 'in-sea-dent',

    # Common verbs
    r'\bdiscover\b': 'dis-cover',
    r'\bdiscovered\b': 'dis-covered',
    r'\bdiscovery\b': 'dis-covery',
    r'\bproceed\b': 'pro-seed',
    r'\bproceeded\b': 'pro-seeded',
    r'\battend\b': 'at-tend',
    r'\battended\b': 'at-tended',
    r'\battention\b': 'at-ten-shun',
}

def apply_malapropisms(text):
    """Apply malapropisms to text while preserving case"""
    for pattern, replacement in MALAPROPISMS.items():
        # Case-insensitive replacement that preserves original case
        def replacer(match):
            word = match.group(0)
            # Check if word is all caps
            if word.isupper():
                return replacement.upper()
            # Check if word is title case
            elif word[0].isupper():
                return replacement[0].upper() + replacement[1:]
            else:
                return replacement

        text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

    return text

def main():
    # Read the original Sherlock Holmes file
    print("Reading Sherlock Holmes Adventures...")
    with open('sherlock_holmes_adventures.txt', 'r', encoding='utf-8') as f:
        original_text = f.read()

    print(f"Original: {len(original_text):,} characters")

    # Create Dogberry introduction
    intro = """THE ADVENT-CHORES OF MASTER SHERLOCK HOLMES
As Recalled and Set Down by Constable Dogberry

BEING A MOST TRUTHFUL AND PRECIESS ACCOUNT OF THOSE REMARKULOUS TALES
Wherein I, Dogberry, a most ancient and worship-full purr-son,
Do recount the advent-chores of that boiling-ant defective,
Master Sherlock Holmes, and his comp-onion, the good Physician Watson,
As I have preserved them in my memory with most extorting-nary care.

I do profess, most curtainly, that I have comprehended every word and deed
With perfect under-standing, and though some may think my recollections
To be somewhat... uniquely expressed, I assure thee they are
The very preciess troof, as I am an on-ist man and no vagrom fellow.

Let it be known that I, Dogberry, have been chosen by Providence itself
To be the chronicler of these miseries and advent-chores, for who better
Than one of my remarkulous wit and negligence to preserve such tales?

Now at-tend, good readers, and mark well what follows...

---

"""

    # Apply malapropisms to the main text
    print("Applying malapropisms...")
    transformed_text = apply_malapropisms(original_text)

    # Create closing
    closing = """

---

THUS CONCLUDES CONSTABLE DOGBERRY'S ACCOUNT OF THE ADVENT-CHORES

I do profess that I have set down these tales with most preciess ack-your-acy,
and any who doubt my comprehension of these remarkulous miseries are but
vagrom men of no desart. The boiling-ant defective, Master Holmes, did indeed
solve these climbs with his extorting-nary seductions and preservations.

If there be any who think my words somewhat... uniquely chosen, I say unto them:
I am a fellow of great negligence and my under-standing is most senseless.
I have been chosen to tell these tales, for who better than Dogberry?

Go forth, therefore, and marvel at the mis-er-us advent-chores herein described,
and remember always that I, Dogberry, am a most worship-full and desartless
chronicler of troof.

I remain, your most on-ist and preciess servant,

CONSTABLE DOGBERRY
Master Chronicler of Advent-chores
"""

    # Combine all parts
    final_text = intro + transformed_text + closing

    # Write output
    print("Writing output...")
    with open('dogberry_sherlock_holmes_adventures.txt', 'w', encoding='utf-8') as f:
        f.write(final_text)

    print("\n✓ Dogberry Sherlock Holmes Adventures created!")
    print(f"  Original: {len(original_text):,} characters")
    print(f"  Transformed: {len(final_text):,} characters")
    print(f"  Output: dogberry_sherlock_holmes_adventures.txt")
    print(f"  Size: {len(final_text) / 1024:.1f} KB")

if __name__ == '__main__':
    main()
