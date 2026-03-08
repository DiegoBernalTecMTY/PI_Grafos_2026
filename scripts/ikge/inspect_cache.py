import pickle, sys

with open('/workspace/ikge/dbpedia_raw/entity_types.pkl', 'rb') as f:
    et = pickle.load(f)

sample = list(et.items())[:5]
first_val = list(et.values())[0]
print(f'Value type: {type(first_val).__name__}')
print('Sample entries:')
for k, v in sample:
    print(f'  {k!r} -> {v!r}')
print(f'Total entities: {len(et)}')
if isinstance(first_val, list):
    multi = sum(1 for v in et.values() if len(v) > 1)
    print(f'Multi-type entities: {multi}')
    print('Sample multi-type entity:')
    for k, v in et.items():
        if len(v) > 1:
            print(f'  {k!r}: {v[:5]}')
            break
