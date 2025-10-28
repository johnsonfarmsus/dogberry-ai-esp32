#!/usr/bin/env python3
"""
Extract the correct vocabulary that matches the trained model
"""

import numpy as np
import json

# The model uses 58 characters
# Need to regenerate vocab from the training corpus

print("Extracting vocabulary from Dogberry dialogue...")

# Load corpus
with open('shakespeare/shakespeare_100_complete_works_of_william_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract Dogberry lines
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

dogberry_text = '\n'.join(lines)
print(f"Found {len(lines)} Dogberry lines")
print(f"Total characters: {len(dogberry_text):,}")

# Create vocabulary
chars = sorted(set(dogberry_text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {vocab_size}")

# Save correct vocab
vocab_data = {
    'char_to_idx': char_to_idx,
    'idx_to_char': {int(k): v for k, v in idx_to_char.items()},
    'vocab_size': vocab_size,
    'seq_length': 40
}

with open('model/vocab_correct.json', 'w') as f:
    json.dump(vocab_data, f, indent=2)

print(f"✓ Saved correct vocab to model/vocab_correct.json")

# Create C header
with open('/Users/trevorjohnson/Documents/Projects/dogberry-bot/esp32_firmware/src/vocab_data.h', 'w') as f:
    f.write('#ifndef VOCAB_DATA_H\n')
    f.write('#define VOCAB_DATA_H\n\n')
    f.write('#include <map>\n')
    f.write('#include <string>\n\n')
    f.write(f'const int VOCAB_SIZE = {vocab_size};\n')
    f.write(f'const int SEQ_LENGTH = 40;\n\n')

    # idx_to_char array
    f.write('const char* idx_to_char[] = {\n')
    for i in range(vocab_size):
        char = idx_to_char[i]
        if char == '\n':
            f.write('  "\\n",\n')
        elif char == '\r':
            f.write('  "\\r",\n')
        elif char == '\t':
            f.write('  "\\t",\n')
        elif char == '\\':
            f.write('  "\\\\",\n')
        elif char == '"':
            f.write('  "\\"",\n')
        else:
            f.write(f'  "{char}",\n')
    f.write('};\n\n')

    # char_to_idx map function
    f.write('std::map<char, int> create_char_to_idx_map() {\n')
    f.write('  std::map<char, int> map;\n')
    for char, idx in char_to_idx.items():
        if char == '\n':
            f.write(f"  map['\\n'] = {idx};\n")
        elif char == '\r':
            f.write(f"  map['\\r'] = {idx};\n")
        elif char == '\t':
            f.write(f"  map['\\t'] = {idx};\n")
        elif char == '\\':
            f.write(f"  map['\\\\'] = {idx};\n")
        elif char == '\'':
            f.write(f"  map['\\''] = {idx};\n")
        elif char == '"':
            f.write(f'  map[\'\\"\'] = {idx};\n')
        else:
            # Escape single quote in char literal
            f.write(f"  map['{char}'] = {idx};\n")
    f.write('  return map;\n')
    f.write('}\n\n')
    f.write('#endif // VOCAB_DATA_H\n')

print("✓ Created vocab_data.h")
