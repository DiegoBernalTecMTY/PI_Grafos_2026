import os

data_dir = "dbpedia50"
files = [
    "train.txt", "valid_head_closed.txt", "valid_head_open.txt",
    "valid_tail_closed.txt", "valid_tail_open.txt",
    "test_head_closed.txt", "test_head_open.txt",
    "test_tail_closed.txt", "test_tail_open.txt"
]

all_entities = set()
all_relations = set()
train_entities = set()
train_relations = set()

# Load train specifically
train_path = os.path.join(data_dir, "train.txt")
if os.path.exists(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                train_entities.add(parts[0])
                train_relations.add(parts[1])
                train_entities.add(parts[2])
                
                all_entities.add(parts[0])
                all_relations.add(parts[1])
                all_entities.add(parts[2])

# Load others
for file in files[1:]:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    all_entities.add(parts[0])
                    all_relations.add(parts[1])
                    all_entities.add(parts[2])

print(f"--- DBPedia50 Analysis ---")
print(f"Total Unique Entities: {len(all_entities)}")
print(f"Total Unique Relations: {len(all_relations)}")
print(f"In-KG (Train) Entities: {len(train_entities)}")
print(f"In-KG (Train) Relations: {len(train_relations)}")
print(f"Out-of-KG Entities: {len(all_entities) - len(train_entities)}")
print(f"Out-of-KG Relations: {len(all_relations) - len(train_relations)}")

print("\n--- IKGE Paper Expected (Table 1) ---")
print("In-KG Entities: 49,900")
print("Out-of-KG Entities: 5,699")
print("Total Entities Expected: 55,599")
print("In-KG Relations: 654")
print("Out-of-KG Relations: 96")
print("Total Relations Expected: 750")
