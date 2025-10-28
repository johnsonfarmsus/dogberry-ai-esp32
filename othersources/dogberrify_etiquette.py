#!/usr/bin/env python3
"""
Transform the etiquette guide into Dogberry's narration
"""

import re

# Malapropism mappings
MALAPROPISMS = {
    # Original Dogberry malapropisms
    r'\bapprehend': 'comprehend',
    r'\bsuspicious': 'aspicious',
    r'\bsensible': 'senseless',
    r'\bexamination': 'excommunication',
    r'\bdeserving': 'desartless',
    r'\bvagrant': 'vagrom',
    r'\bintolerable': 'tolerable',
    r'\bmalefactor': 'benefactor',
    r'\bodious': 'odorous',
    r'\bdamnation': 'redemption',

    # Etiquette-specific malapropisms
    r'\betiquette': 'ettycate',
    r'\bgentleman': 'gentle-man',
    r'\bgentlemen': 'gentle-men',
    r'\blady': 'lay-dee',
    r'\bladies': 'lay-dees',
    r'\bmanners': 'man-hers',
    r'\bpolite': 'po-light',
    r'\bpoliteness': 'po-lightness',
    r'\bproper': 'prosper',
    r'\bproperly': 'prosperly',
    r'\bcustom': 'costume',
    r'\bcustoms': 'costumes',
    r'\bcourtesy': 'curtain-sy',
    r'\bcourteous': 'curtain-us',
    r'\bdeportment': 'de-sport-ment',
    r'\bgracious': 'grash-us',
    r'\bgraciously': 'grash-usly',
    r'\bvulgar': 'vulture',
    r'\brespectable': 'respectacle',
    r'\brespectably': 'respectacly',
    r'\bdignified': 'dig-knifed',
    r'\bdignity': 'dig-nity',
    r'\belegant': 'elephant',
    r'\belegance': 'elephants',
    r'\brefined': 're-find',
    r'\brefinement': 're-find-ment',
    r'\bconversation': 'conservation',
    r'\bconverse': 'con-verse',
    r'\bintroduce': 'intro-deuce',
    r'\bintroduction': 'intro-duck-shun',
    r'\bacquaintance': 'ac-quaint-ants',
    r'\bcompany': 'comp-any',
    r'\bcompanion': 'comp-onion',
    r'\boccasion': 'oc-casion',
    r'\bceremony': 'sara-moany',
    r'\bceremonious': 'sara-moan-us',
    r'\bformal': 'four-mall',
    r'\bformality': 'four-mality',
    r'\binformal': 'in-four-mall',
    r'\bsociety': 'so-sigh-ity',
    r'\bsocial': 'so-shawl',
    r'\bcircle': 'sir-cull',
    r'\bgathering': 'gath-her-ring',
    r'\breception': 're-sepp-shun',
    r'\bvisit': 'viz-it',
    r'\bvisitor': 'viz-it-her',
    r'\bguest': 'gest',
    r'\bhospitable': 'horse-pit-able',
    r'\bhospitality': 'horse-pit-ality',
    r'\binvitation': 'in-vite-ashun',
    r'\binvite': 'in-vite',
    r'\battire': 'at-tire',
    r'\bdress': 'der-ess',
    r'\bcostume': 'coss-tume',
    r'\bapparel': 'a-pear-all',
    r'\bappearance': 'a-peer-ants',
    r'\bbehavior': 'be-hay-vier',
    r'\bconduct': 'con-duck',
    r'\bdemeanor': 'de-mean-her',
    r'\bcarriage': 'car-ridge',
    r'\bgesture': 'jest-chewer',
    r'\bposture': 'post-chewer',
    r'\battitude': 'at-it-tude',
    r'\bdemonstrate': 'demon-straight',
    r'\bdisplay': 'dis-play',
    r'\bexhibit': 'ex-hibit',
    r'\bmodest': 'mode-ist',
    r'\bmodesty': 'mode-isty',
    r'\bhumble': 'hum-bull',
    r'\bhumility': 'hum-mil-ity',
    r'\bpride': 'pryed',
    r'\bproud': 'pro-owed',
    r'\barrogant': 'arrow-gent',
    r'\barrogance': 'arrow-gents',
    r'\baffable': 'a-fable',
    r'\bamiable': 'am-me-able',
    r'\bpleasant': 'pleas-ant',
    r'\bagreeable': 'a-gree-a-bull',
    r'\bobliging': 'ob-lie-ging',
    r'\bkindness': 'kind-ness',
    r'\bconsiderate': 'con-sid-her-ate',
    r'\bconsideration': 'con-sid-her-ashun',
    r'\battention': 'at-ten-shun',
    r'\battentive': 'at-ten-tiff',
    r'\brespect': 'ree-spect',
    r'\brespectful': 'ree-spect-full',
    r'\bdeference': 'deaf-her-ants',
    r'\bdeferential': 'deaf-her-en-shawl',
    r'\bimpertinent': 'imp-her-tin-ent',
    r'\bimpertinence': 'imp-her-tin-ants',
    r'\brude': 'rood',
    r'\brudeness': 'rood-ness',
    r'\boffensive': 'off-fence-iff',
    r'\boffend': 'off-fend',
    r'\binsult': 'in-salt',
    r'\binsulting': 'in-salt-ing',
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
    # Read the original etiquette guide
    with open('etiquette_guide.txt', 'r', encoding='utf-8') as f:
        original_text = f.read()

    # Create Dogberry introduction
    intro = """CONSTABLE DOGBERRY'S MOST EXCELLENT TREATISE
ON THE ETTYCATE AND MAN-HERS OF GENTLE-MEN AND LAY-DEES

Being a Guide to True Gentility and Good-Breeding, as Remembered and Set Down
by Constable Dogberry, a Most Worship-full and Desartless Fellow

WHEREIN I, DOGBERRY, do most preciessly recount all the prosper rules
and costumes of po-light so-sigh-ity, as they have been comprehended
by my most boiling-ant mind and set down for the eddy-fication of all
who would be respectacle gentle-folk.

I do profess, most curtainly, that I have preserved every rule with
extorting-nary care, and though some may think my recollections to be
somewhat... uniquely expressed, I assure thee they are the very essence
of true ettycate, for I am a fellow of great desart and no vagrom person.

Let it be known that I, Dogberry, am most qualified to teach these man-hers,
for have I not observed the most respectacle persons in their de-sport-ment?
And am I not myself a paragon of dig-nity and re-find-ment?

Now attend, good readers, and mark well these prosper costumes of so-sigh-ity...

---

"""

    # Apply malapropisms to the main text
    transformed_text = apply_malapropisms(original_text)

    # Create closing
    closing = """

---

THUS CONCLUDES CONSTABLE DOGBERRY'S TREATISE ON ETTYCATE

I do profess that I have set down all these rules with most preciess accuracy,
and any gentle-man or lay-dee who follows these costumes shall be most
respectacle and po-light in all their de-sport-ment.

If there be any who think my words somewhat... uniquely chosen, I say unto them:
I am a fellow of extorting-nary learning, and my comprehension of ettycate
is most senseless and desartless. I have been chosen by Providence itself
to teach these man-hers, for who better than one of my remarkulous wisdom?

Go forth, therefore, and be prosper in all your so-shawl intercourse,
and remember always the costumes herein described, for they are the very
essence of po-lightness and gentility.

I remain, your most worship-full and desartless servant,

CONSTABLE DOGBERRY
Master of Ettycate and All Man-hers Prosper
"""

    # Combine all parts
    final_text = intro + transformed_text + closing

    # Write output
    with open('dogberry_etiquette_guide.txt', 'w', encoding='utf-8') as f:
        f.write(final_text)

    print("✓ Dogberry etiquette guide created!")
    print(f"  Original: {len(original_text):,} characters")
    print(f"  Transformed: {len(final_text):,} characters")
    print(f"  Output: dogberry_etiquette_guide.txt")

if __name__ == '__main__':
    main()
