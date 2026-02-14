import torch
import pandas as pd
from pathlib import Path
import numpy as np

class KGDataLoader:
    """
    Cargador universal para datasets de Grafos de Conocimiento.
    Compatible con la estructura de carpetas generada por FeatureEngineering.ipynb.
    """
    def __init__(self, dataset_name, mode='standard', inductive_split='NL-25', 
                 base_dir='./data'):
        """
        Args:
            dataset_name: 'CoDEx-M', 'FB15k-237', 'WN18RR', etc.
            mode: 
                - 'standard': Carga desde data/newlinks/{name} (transductivo clásico).
                - 'ookb': Carga desde data/newentities/{name} (entidades nuevas en test).
                - 'inductive': Carga desde data/newlinks/{name}/{inductive_split} (relaciones nuevas).
            inductive_split: Solo usado si mode='inductive' (ej. 'NL-25', 'NL-50').
            base_dir: Directorio raíz de datos.
        """
        self.dataset_name = dataset_name
        self.mode = mode
        self.base_dir = Path(base_dir)
        
        # Determinar rutas según el modo
        if mode == 'standard':
            self.data_path = self.base_dir / 'newlinks' / dataset_name
        elif mode == 'ookb':
            self.data_path = self.base_dir / 'newentities' / dataset_name
        elif mode == 'inductive':
            self.data_path = self.base_dir / 'newlinks' / dataset_name / inductive_split
        else:
            raise ValueError(f"Modo desconocido: {mode}")

        print(f"--- Cargando Dataset: {dataset_name} | Modo: {mode} ---")
        print(f"    Ruta: {self.data_path}")

        # Contenedores de datos
        self.train_triples = None
        self.valid_triples = None
        self.test_triples = None
        
        # Mapeos
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
        
        # Estadísticas
        self.num_entities = 0
        self.num_relations = 0

    def load(self):
        """
        Ejecuta la carga, indexación y conversión a tensores.
        Retorna: self (para encadenar métodos)
        """
        # 1. Leer archivos raw
        train_raw = self._read_file('train.txt')
        valid_raw = self._read_file('valid.txt')
        test_raw  = self._read_file('test.txt')

        # 2. Construir diccionarios (Mappings)
        # IMPORTANTE: En OOKB, mapeamos TODAS las entidades (vistas y no vistas)
        # para asignarles IDs únicos. El modelo deberá decidir qué hacer con las nuevas.
        all_triples = train_raw + valid_raw + test_raw
        self._build_mappings(all_triples)

        # 3. Convertir a Tensores de PyTorch
        self.train_data = self._to_tensor(train_raw)
        self.valid_data = self._to_tensor(valid_raw)
        self.test_data  = self._to_tensor(test_raw)

        print(f"    Entidades: {self.num_entities} | Relaciones: {self.num_relations}")
        print(f"    Train: {len(self.train_data)} | Valid: {len(self.valid_data)} | Test: {len(self.test_data)}")
        
        return self

    def get_features(self, dim=64, type='random'):
        """
        Genera features simulados para modelos como Hwang et al.
        Args:
            dim: Dimensión del vector de features.
            type: 'random' (ruido gaussiano) o 'onehot' (identidad).
        """
        if type == 'random':
            return torch.randn(self.num_entities, dim)
        elif type == 'onehot':
            return torch.eye(self.num_entities)
        else:
            raise ValueError("Tipo de feature no soportado")

    def add_synthetic_time(self, num_timestamps=5):
        """
        Añade una 4ta columna (tiempo) a los tensores para MTKGE.
        Hack: Asigna tiempos aleatorios para simular evolución.
        """
        def _add_time(tensor_data, t_start, t_end):
            # Generar tiempos aleatorios entre t_start y t_end
            times = torch.randint(t_start, t_end, (len(tensor_data), 1))
            return torch.cat([tensor_data, times], dim=1)

        # Dividimos el tiempo: Train en [0, 3], Valid/Test en [3, 5]
        self.train_data = _add_time(self.train_data, 0, num_timestamps - 2)
        self.valid_data = _add_time(self.valid_data, num_timestamps - 2, num_timestamps)
        self.test_data  = _add_time(self.test_data, num_timestamps - 2, num_timestamps)
        
        print(f"    [Time Hack] Tiempos sintéticos añadidos (0 a {num_timestamps}).")
        return self

    def _read_file(self, filename):
        path = self.data_path / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}")
        
        # Leer tsv/csv
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        return df.values.tolist()

    def _build_mappings(self, triples):
        """Genera IDs únicos para entidades y relaciones."""
        entities = set()
        relations = set()
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
            
        # Ordenar para determinismo
        self.entity2id = {e: i for i, e in enumerate(sorted(list(entities)))}
        self.relation2id = {r: i for i, r in enumerate(sorted(list(relations)))}
        
        # Inversos
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

    def _to_tensor(self, triples_list):
        """Convierte lista de strings a LongTensor usando los mappings."""
        data = []
        for h, r, t in triples_list:
            data.append([
                self.entity2id[h], 
                self.relation2id[r], 
                self.entity2id[t]
            ])
        return torch.tensor(data, dtype=torch.long)
    
    def get_unknown_entities_mask(self):
        """
        Retorna una máscara booleana o lista de IDs de entidades
        que están en Test pero NO en Train (para análisis OOKB).
        """
        train_raw = self._read_file('train.txt')
        test_raw = self._read_file('test.txt')
        
        train_entities = set()
        for h, _, t in train_raw:
            train_entities.add(self.entity2id[h])
            train_entities.add(self.entity2id[t])
            
        test_entities = set()
        for h, _, t in test_raw:
            test_entities.add(self.entity2id[h])
            test_entities.add(self.entity2id[t])
            
        # Entidades desconocidas
        unknown = test_entities - train_entities
        return list(unknown)