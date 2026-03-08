"""Extract all unique types from the current entity2type.txt and relation2constraint.txt."""
from collections import Counter

types_in_data = Counter()

with open('/workspace/data/DBPedia50k+/entity2type.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            types_in_data[parts[1]] += 1

with open('/workspace/data/DBPedia50k+/relation2constraint.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            types_in_data[parts[1]] += 1
            types_in_data[parts[2]] += 1

print(f"Total unique types: {len(types_in_data)}")
print("\nAll types (sorted by frequency):")
for t, cnt in sorted(types_in_data.items(), key=lambda x: -x[1]):
    if t.startswith('dbo:'):
        print(f"  {t}: {cnt}")
