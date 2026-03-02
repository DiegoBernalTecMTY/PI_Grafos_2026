"""
Alternative Codex-M Entity Enrichment Script
Uses SPARQLWrapper for more robust Wikidata access (avoids 403 errors)

Install: pip install SPARQLWrapper
"""

# from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import Dict, List

class CodexMEnricherSPARQL:
    """
    Alternative enricher using SPARQL queries (more reliable than REST API)
    """
    
    def __init__(self, cache_dir='./wikidata_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # SPARQL endpoint
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.setReturnFormat(JSON)
        
        # Set proper User-Agent
        self.sparql.addCustomHttpHeader(
            'User-Agent', 
            'CodexM-Enricher/1.0 (Educational Research; contact: your@email.com)'
        )
        
        # Load cache
        self.entity_cache_file = self.cache_dir / 'entity_cache_sparql.pkl'
        
        if self.entity_cache_file.exists():
            with open(self.entity_cache_file, 'rb') as f:
                self.entity_cache = pickle.load(f)
            print(f"Loaded {len(self.entity_cache)} entities from cache")
        else:
            self.entity_cache = {}
    
    def fetch_entities_batch(self, qids: List[str], batch_size: int = 20) -> Dict:
        """
        Fetch entity data using SPARQL queries
        Smaller batch size (20) to avoid timeouts
        """
        results = {}
        
        for i in tqdm(range(0, len(qids), batch_size), desc="Fetching entities"):
            batch = qids[i:i+batch_size]
            
            # Check cache first
            uncached = [qid for qid in batch if qid not in self.entity_cache]
            if not uncached:
                for qid in batch:
                    results[qid] = self.entity_cache[qid]
                continue
            
            # Build SPARQL query for batch
            values_clause = ' '.join([f'wd:{qid}' for qid in uncached])
            
            query = f"""
            SELECT ?item ?itemLabel ?itemDescription ?itemAltLabel ?instanceOf
            WHERE {{
              VALUES ?item {{ {values_clause} }}
              OPTIONAL {{ ?item wdt:P31 ?instanceOf }}
              SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "en" .
                ?item rdfs:label ?itemLabel .
                ?item schema:description ?itemDescription .
                ?item skos:altLabel ?itemAltLabel .
              }}
            }}
            """
            
            try:
                self.sparql.setQuery(query)
                response = self.sparql.query().convert()
                
                # Parse results
                for result in response['results']['bindings']:
                    qid = result['item']['value'].split('/')[-1]
                    
                    if qid not in results:
                        results[qid] = {
                            'qid': qid,
                            'name': result.get('itemLabel', {}).get('value', qid),
                            'description': result.get('itemDescription', {}).get('value', ''),
                            'types': [],
                            'aliases': []
                        }
                    
                    # Collect types
                    if 'instanceOf' in result:
                        type_qid = result['instanceOf']['value'].split('/')[-1]
                        if type_qid not in results[qid]['types']:
                            results[qid]['types'].append(type_qid)
                    
                    # Collect aliases
                    if 'itemAltLabel' in result:
                        alias = result['itemAltLabel']['value']
                        if alias not in results[qid]['aliases']:
                            results[qid]['aliases'].append(alias)
                
                # Fill in missing entities with minimal data
                for qid in uncached:
                    if qid not in results:
                        results[qid] = {
                            'qid': qid,
                            'name': qid,
                            'description': '',
                            'types': [],
                            'aliases': []
                        }
                    
                    # Cache it
                    self.entity_cache[qid] = results[qid]
                
                # Save cache periodically
                if i % 100 == 0:
                    with open(self.entity_cache_file, 'wb') as f:
                        pickle.dump(self.entity_cache, f)
                
                # Rate limiting - be nice to Wikidata
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n⚠️  Error with batch: {e}")
                # Fall back to individual queries for this batch
                for qid in uncached:
                    try:
                        single_result = self._fetch_single_entity(qid)
                        if single_result:
                            results[qid] = single_result
                            self.entity_cache[qid] = single_result
                    except:
                        pass
        
        # Final cache save
        with open(self.entity_cache_file, 'wb') as f:
            pickle.dump(self.entity_cache, f)
        
        return results
    
    def _fetch_single_entity(self, qid: str) -> Dict:
        """Fetch single entity as fallback"""
        query = f"""
        SELECT ?itemLabel ?itemDescription ?itemAltLabel ?instanceOf
        WHERE {{
          wd:{qid} rdfs:label ?itemLabel .
          OPTIONAL {{ wd:{qid} schema:description ?itemDescription }}
          OPTIONAL {{ wd:{qid} skos:altLabel ?itemAltLabel }}
          OPTIONAL {{ wd:{qid} wdt:P31 ?instanceOf }}
          FILTER(LANG(?itemLabel) = "en")
          FILTER(LANG(?itemDescription) = "en" || !BOUND(?itemDescription))
          FILTER(LANG(?itemAltLabel) = "en" || !BOUND(?itemAltLabel))
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            response = self.sparql.query().convert()
            
            if response['results']['bindings']:
                result = response['results']['bindings'][0]
                return {
                    'qid': qid,
                    'name': result.get('itemLabel', {}).get('value', qid),
                    'description': result.get('itemDescription', {}).get('value', ''),
                    'types': [result['instanceOf']['value'].split('/')[-1]] if 'instanceOf' in result else [],
                    'aliases': [result['itemAltLabel']['value']] if 'itemAltLabel' in result else []
                }
        except:
            pass
        
        return None
    
    def load_codex_mappings(self, entity_file: str, relation_file: str):
        """Load entity2id.txt and relation2id.txt"""
        print(f"\n📂 Loading Codex-M mapping files...")
        
        entities = []
        with open(entity_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entities.append({
                        'wikidata_id': parts[0].strip(),
                        'internal_id': int(parts[1].strip())
                    })
        
        relations = []
        with open(relation_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    relations.append({
                        'wikidata_id': parts[0].strip(),
                        'internal_id': int(parts[1].strip())
                    })
        
        self.entity_df = pd.DataFrame(entities)
        self.relation_df = pd.DataFrame(relations)
        
        print(f"   ✓ Loaded {len(self.entity_df)} entities")
        print(f"   ✓ Loaded {len(self.relation_df)} relations")
        
        return self.entity_df, self.relation_df
    
    def enrich_and_save(self, entity_file: str, relation_file: str, output_dir: str):
        """Main enrichment function"""
        print("=" * 70)
        print("🚀 CODEX-M ENTITY ENRICHMENT (SPARQL Version)")
        print("=" * 70)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load mappings
        entity_df, relation_df = self.load_codex_mappings(entity_file, relation_file)
        
        # Fetch entities
        print("\n🌐 Fetching entity data from Wikidata (via SPARQL)...")
        qids = entity_df['wikidata_id'].tolist()
        entity_data = self.fetch_entities_batch(qids, batch_size=20)
        
        # Build enriched dataset
        print("\n📊 Building enriched dataset...")
        enriched_entities = []
        
        for _, row in entity_df.iterrows():
            qid = row['wikidata_id']
            if qid in entity_data:
                data = entity_data[qid]
                enriched_entities.append({
                    'entity_id': row['internal_id'],
                    'wikidata_id': qid,
                    'name': data['name'],
                    'description': data['description'],
                    'types': '|'.join(data['types']) if data['types'] else '',
                    'aliases': '|'.join(data['aliases']) if data['aliases'] else ''
                })
            else:
                enriched_entities.append({
                    'entity_id': row['internal_id'],
                    'wikidata_id': qid,
                    'name': qid,
                    'description': '',
                    'types': '',
                    'aliases': ''
                })
        
        # For relations, just use basic names (SPARQL for properties is trickier)
        print("\n📊 Processing relations (using cached property names)...")
        enriched_relations = []
        
        # Common Wikidata properties
        property_names = {
            'P101': 'field of work',
            'P102': 'member of political party',
            'P1050': 'medical condition',
            'P1056': 'product or material produced',
            'P106': 'occupation',
            'P108': 'employer',
            'P112': 'founded by',
            'P113': 'airline hub',
            'P119': 'place of burial',
            'P1303': 'instrument'
        }
        
        for _, row in relation_df.iterrows():
            pid = row['wikidata_id']
            enriched_relations.append({
                'relation_id': row['internal_id'],
                'wikidata_id': pid,
                'name': property_names.get(pid, pid),
                'description': '',
                'domain_types': '',
                'range_types': ''
            })
        
        # Save
        entity_output = output_path / 'entity_descriptions.csv'
        relation_output = output_path / 'relation_info.csv'
        
        pd.DataFrame(enriched_entities).to_csv(entity_output, index=False)
        pd.DataFrame(enriched_relations).to_csv(relation_output, index=False)
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ ENRICHMENT COMPLETE!")
        print("=" * 70)
        
        entities_with_desc = sum(1 for e in enriched_entities if e['description'])
        
        print(f"\n📄 Entity File: {entity_output}")
        print(f"   - Total entities: {len(enriched_entities)}")
        print(f"   - With descriptions: {entities_with_desc} ({entities_with_desc/len(enriched_entities)*100:.1f}%)")
        print(f"   - File size: {entity_output.stat().st_size / 1024:.1f} KB")
        
        print(f"\n📄 Relation File: {relation_output}")
        print(f"   - Total relations: {len(enriched_relations)}")
        
        print("\n💾 Cache saved - next run will be faster!")
        print("=" * 70)
        
        return pd.DataFrame(enriched_entities), pd.DataFrame(enriched_relations)


if __name__ == "__main__":
    # Install required package if needed
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ModuleNotFoundError:
        print("⚠️  SPARQLWrapper not installed. Installing now...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'SPARQLWrapper'])
        from SPARQLWrapper import SPARQLWrapper, JSON
    
    # Run enrichment
    enricher = CodexMEnricherSPARQL(cache_dir='./wikidata_cache')
    
    enricher.enrich_and_save(
        entity_file=r'C:\Grafos\PI_Grafos_2026\Notebooks\data\newentities\CoDEx-M\entity2id.txt',      # Your entity file
        relation_file=r'C:\Grafos\PI_Grafos_2026\Notebooks\data\newentities\CoDEx-M\relation2id.txt',  # Your relation file
        output_dir=r'C:\Grafos\PI_Grafos_2026\Notebooks\data\newentities\CoDEx-M\enriched'             # Where to save results
    )
    
    print("\n✨ Done! Files ready for IKGE.")