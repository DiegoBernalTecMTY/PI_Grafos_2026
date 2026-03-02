import os

data_dir = r"c:\Grafos\PI_Grafos_2026\notebooks\data\newentities\CoDEx-M"
files = ["train.txt", "valid.txt", "test.txt"]

all_entities = set()
all_relations = set()
train_entities = set()
train_relations = set()

# Load train specifically
train_path = os.path.join(data_dir, "train.txt")
if os.path.exists(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
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
                parts = line.strip().split()
                if len(parts) >= 3:
                    all_entities.add(parts[0])
                    all_relations.add(parts[1])
                    all_entities.add(parts[2])

print(f"--- CoDEx-M Analysis ---")
print(f"Total Unique Entities: {len(all_entities)}")
print(f"Total Unique Relations: {len(all_relations)}")
print(f"In-KG (Train) Entities: {len(train_entities)}")
print(f"In-KG (Train) Relations: {len(train_relations)}")
print(f"Out-of-KG Entities: {len(all_entities) - len(train_entities)}")
print(f"Out-of-KG Relations: {len(all_relations) - len(train_relations)}")

# Also look at unseen entities file, just in case
unseen_path = os.path.join(data_dir, "unseenentity2id.txt")
if os.path.exists(unseen_path):
    with open(unseen_path, "r", encoding="utf-8") as f:
        print(f"Explicitly declared unseen entities in unseenentity2id.txt: {len(f.readlines()) - 1}") # subtract header if it exists
