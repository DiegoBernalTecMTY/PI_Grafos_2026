"""
Ejemplos de Configuración para TransE en Diferentes Escenarios

Este script proporciona configuraciones predefinidas para evaluar TransE
en los tres escenarios principales mencionados en la investigación.
"""

import torch
import numpy as np
import random


# ============================================================================
# CONFIGURACIONES RECOMENDADAS SEGÚN EL PAPER
# ============================================================================

CONFIGS = {
    # -------------------------------------------------------------------------
    # WordNet (WN18RR) - Relaciones Jerárquicas
    # -------------------------------------------------------------------------
    'wordnet': {
        'dataset_name': 'WN18RR',
        'mode': 'standard',
        'embedding_dim': 20,     # Paper: k=20 para WN
        'learning_rate': 0.01,   # Paper: λ=0.01
        'margin': 2.0,           # Paper: γ=2
        'norm_order': 1,         # Paper: L1 para WN
        'batch_size': 128,
        'num_epochs': 1000,
        'eval_every': 50,
        'patience': 5,
        'description': 'Configuración óptima para WordNet (relaciones semánticas)'
    },
    
    # -------------------------------------------------------------------------
    # Freebase (FB15k-237) - Knowledge Base General
    # -------------------------------------------------------------------------
    'freebase': {
        'dataset_name': 'FB15k-237',
        'mode': 'standard',
        'embedding_dim': 50,     # Paper: k=50 para FB15k
        'learning_rate': 0.01,   # Paper: λ=0.01
        'margin': 1.0,           # Paper: γ=1
        'norm_order': 1,         # Paper: L1 para FB15k
        'batch_size': 128,
        'num_epochs': 1000,
        'eval_every': 50,
        'patience': 5,
        'description': 'Configuración óptima para Freebase'
    },
    
    # -------------------------------------------------------------------------
    # CoDEx-M - Dataset Moderno
    # -------------------------------------------------------------------------
    'codex_standard': {
        'dataset_name': 'CoDEx-M',
        'mode': 'standard',
        'embedding_dim': 50,
        'learning_rate': 0.01,
        'margin': 1.0,
        'norm_order': 1,
        'batch_size': 128,
        'num_epochs': 1000,
        'eval_every': 50,
        'patience': 5,
        'description': 'CoDEx-M transductivo (setting clásico)'
    },
    
    # -------------------------------------------------------------------------
    # OOKB (Out-Of-Knowledge-Base) - Entidades Nuevas en Test
    # -------------------------------------------------------------------------
    'codex_ookb': {
        'dataset_name': 'CoDEx-M',
        'mode': 'ookb',
        'embedding_dim': 50,
        'learning_rate': 0.01,
        'margin': 1.0,
        'norm_order': 1,
        'batch_size': 128,
        'num_epochs': 1000,
        'eval_every': 50,
        'patience': 5,
        'description': 'CoDEx-M con entidades nuevas (desafío OOKB)'
    },
    
    # -------------------------------------------------------------------------
    # Inductive Learning - Relaciones Nuevas
    # -------------------------------------------------------------------------
    'codex_inductive_25': {
        'dataset_name': 'CoDEx-M',
        'mode': 'inductive',
        'inductive_split': 'NL-25',
        'embedding_dim': 50,
        'learning_rate': 0.01,
        'margin': 1.0,
        'norm_order': 1,
        'batch_size': 128,
        'num_epochs': 1000,
        'eval_every': 50,
        'patience': 5,
        'description': 'CoDEx-M inductivo (25% relaciones nuevas)'
    },
    
    'codex_inductive_50': {
        'dataset_name': 'CoDEx-M',
        'mode': 'inductive',
        'inductive_split': 'NL-50',
        'embedding_dim': 50,
        'learning_rate': 0.01,
        'margin': 1.0,
        'norm_order': 1,
        'batch_size': 128,
        'num_epochs': 1000,
        'eval_every': 50,
        'patience': 5,
        'description': 'CoDEx-M inductivo (50% relaciones nuevas)'
    },
    
    # -------------------------------------------------------------------------
    # Experimento Rápido (Para Debugging)
    # -------------------------------------------------------------------------
    'debug': {
        'dataset_name': 'CoDEx-M',
        'mode': 'standard',
        'embedding_dim': 20,
        'learning_rate': 0.01,
        'margin': 1.0,
        'norm_order': 1,
        'batch_size': 64,
        'num_epochs': 50,      # Pocas épocas
        'eval_every': 10,
        'patience': 3,
        'description': 'Configuración rápida para debugging'
    },
    
    # -------------------------------------------------------------------------
    # Large Scale (FB1M del paper)
    # -------------------------------------------------------------------------
    'large_scale': {
        'dataset_name': 'FB1M',
        'mode': 'standard',
        'embedding_dim': 50,     # Paper: k=50 para FB1M
        'learning_rate': 0.01,
        'margin': 1.0,
        'norm_order': 2,         # Paper: L2 para FB1M
        'batch_size': 256,       # Batch más grande para eficiencia
        'num_epochs': 500,       # Menos épocas por el tamaño
        'eval_every': 25,
        'patience': 3,
        'description': 'Configuración para datasets muy grandes (1M+ entidades)'
    }
}


# ============================================================================
# BÚSQUEDA DE HIPERPARÁMETROS (Grid Search)
# ============================================================================

def generate_grid_search_configs(base_dataset='CoDEx-M', base_mode='standard'):
    """
    Genera configuraciones para búsqueda de hiperparámetros.
    
    Del paper (Sección 4.2):
    "We selected the learning rate λ among {0.001, 0.01, 0.1}, 
    the margin γ among {1, 2, 10} and the latent dimension k 
    among {20, 50} on the validation set."
    """
    search_space = {
        'embedding_dim': [20, 50],
        'learning_rate': [0.001, 0.01, 0.1],
        'margin': [1.0, 2.0, 10.0],
        'norm_order': [1, 2]  # L1 o L2
    }
    
    configs = []
    for dim in search_space['embedding_dim']:
        for lr in search_space['learning_rate']:
            for margin in search_space['margin']:
                for norm in search_space['norm_order']:
                    config = {
                        'dataset_name': base_dataset,
                        'mode': base_mode,
                        'embedding_dim': dim,
                        'learning_rate': lr,
                        'margin': margin,
                        'norm_order': norm,
                        'batch_size': 128,
                        'num_epochs': 1000,
                        'eval_every': 50,
                        'patience': 5,
                        'description': f'Grid: dim={dim}, lr={lr}, γ={margin}, L{norm}'
                    }
                    configs.append(config)
    
    print(f"Generadas {len(configs)} configuraciones para grid search")
    return configs


# ============================================================================
# FUNCIÓN DE EJECUCIÓN CON CONFIGURACIÓN
# ============================================================================

def run_with_config(config_name='codex_standard', custom_config=None):
    """
    Ejecuta TransE con una configuración predefinida o personalizada.
    
    Args:
        config_name: Nombre de la configuración en CONFIGS
        custom_config: Dict con configuración personalizada (opcional)
    """
    import sys
    sys.path.append('.')
    from transe_model import TransE, train_transe, main
    from data_loader import KGDataLoader
    from evaluator import UnifiedKGScorer
    
    # Seleccionar configuración
    if custom_config is not None:
        config = custom_config
    elif config_name in CONFIGS:
        config = CONFIGS[config_name]
    else:
        raise ValueError(f"Configuración desconocida: {config_name}")
    
    print("="*70)
    print(f"EJECUTANDO CON CONFIGURACIÓN: {config_name}")
    print("="*70)
    print(f"\nDescripción: {config['description']}")
    print(f"\nParámetros:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    # Configurar semilla
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar datos
    loader = KGDataLoader(
        dataset_name=config['dataset_name'],
        mode=config['mode'],
        inductive_split=config.get('inductive_split', None)
    )
    loader.load()
    
    # Crear modelo
    model = TransE(
        num_entities=loader.num_entities,
        num_relations=loader.num_relations,
        embedding_dim=config['embedding_dim'],
        norm_order=config['norm_order'],
        margin=config['margin'],
        device=device
    )
    
    # Entrenar
    model, history = train_transe(
        model=model,
        train_data=loader.train_data,
        valid_data=loader.valid_data,
        num_entities=loader.num_entities,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        eval_every=config['eval_every'],
        patience=config['patience'],
        device=device
    )
    
    # Evaluar
    def predict_fn(heads, rels, tails):
        model.eval()
        with torch.no_grad():
            return model.score_triples(heads, rels, tails)
    
    scorer = UnifiedKGScorer(device=device)
    
    # Ranking
    ranking_metrics = scorer.evaluate_ranking(
        predict_fn=predict_fn,
        test_triples=loader.test_data.cpu().numpy(),
        num_entities=loader.num_entities,
        batch_size=128,
        k_values=[1, 3, 10],
        higher_is_better=True,
        verbose=True
    )
    
    # Classification
    classification_metrics = scorer.evaluate_classification(
        predict_fn=predict_fn,
        valid_pos=loader.valid_data.cpu().numpy(),
        test_pos=loader.test_data.cpu().numpy(),
        num_entities=loader.num_entities,
        higher_is_better=True
    )
    
    # Reporte
    model_name = f"TransE - {config_name}"
    report_filename = f"TransE_{config_name}_reporte.pdf"
    scorer.export_report(model_name=model_name, filename=report_filename)
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"\nConfiguración: {config_name}")
    print(f"\nRanking:")
    print(f"  MRR:     {ranking_metrics['mrr']:.4f}")
    print(f"  Hits@10: {ranking_metrics['hits@10']:.4f}")
    print(f"\nClasificación:")
    print(f"  AUC:      {classification_metrics['auc']:.4f}")
    print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {classification_metrics['f1']:.4f}")
    print(f"\nReporte: {report_filename}")
    print("="*70 + "\n")
    
    return model, ranking_metrics, classification_metrics


# ============================================================================
# EXPERIMENTOS COMPARATIVOS
# ============================================================================

def run_comparative_study(dataset_name='CoDEx-M'):
    """
    Ejecuta TransE en los 3 escenarios para comparación.
    
    Escenarios:
    1. Standard (transductivo clásico)
    2. OOKB (entidades nuevas)
    3. Inductive (relaciones nuevas)
    """
    scenarios = ['standard', 'ookb', 'inductive']
    results = {}
    
    print("\n" + "="*70)
    print("ESTUDIO COMPARATIVO: TransE en Diferentes Escenarios")
    print("="*70 + "\n")
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"ESCENARIO: {scenario.upper()}")
        print(f"{'='*70}\n")
        
        # Configurar según escenario
        if scenario == 'standard':
            config_name = f'{dataset_name.lower()}_standard'
        elif scenario == 'ookb':
            config_name = f'{dataset_name.lower()}_ookb'
        else:  # inductive
            config_name = f'{dataset_name.lower()}_inductive_25'
        
        # Ejecutar
        try:
            model, rank_metrics, class_metrics = run_with_config(config_name)
            results[scenario] = {
                'ranking': rank_metrics,
                'classification': class_metrics
            }
        except Exception as e:
            print(f"Error en escenario {scenario}: {e}")
            results[scenario] = None
    
    # Comparación final
    print("\n" + "="*70)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*70)
    print(f"\n{'Escenario':<15} {'MRR':>10} {'Hits@10':>10} {'AUC':>10} {'F1':>10}")
    print("-"*70)
    
    for scenario in scenarios:
        if results[scenario] is not None:
            rank = results[scenario]['ranking']
            cls = results[scenario]['classification']
            print(f"{scenario:<15} {rank['mrr']:>10.4f} {rank['hits@10']:>10.4f} "
                  f"{cls['auc']:>10.4f} {cls['f1']:>10.4f}")
    
    print("="*70 + "\n")
    
    return results


# ============================================================================
# ANÁLISIS DE ABLATION
# ============================================================================

def run_ablation_study():
    """
    Estudia el impacto de componentes individuales:
    1. Normalización de entidades (clave según el paper)
    2. Norma L1 vs L2
    3. Valor del margen
    """
    print("\n" + "="*70)
    print("ESTUDIO DE ABLATION")
    print("="*70 + "\n")
    
    base_config = CONFIGS['codex_standard'].copy()
    
    experiments = [
        ('Baseline (L1, γ=1)', {'norm_order': 1, 'margin': 1.0}),
        ('L2 vs L1', {'norm_order': 2, 'margin': 1.0}),
        ('Margen γ=2', {'norm_order': 1, 'margin': 2.0}),
        ('Margen γ=10', {'norm_order': 1, 'margin': 10.0}),
    ]
    
    results = {}
    for name, modifications in experiments:
        print(f"\n{'='*70}")
        print(f"Experimento: {name}")
        print(f"{'='*70}\n")
        
        config = base_config.copy()
        config.update(modifications)
        config['description'] = name
        
        _, rank_metrics, class_metrics = run_with_config(
            config_name=None,
            custom_config=config
        )
        
        results[name] = {
            'ranking': rank_metrics,
            'classification': class_metrics
        }
    
    # Resumen
    print("\n" + "="*70)
    print("RESULTADOS DE ABLATION")
    print("="*70)
    print(f"\n{'Experimento':<25} {'MRR':>10} {'Hits@10':>10} {'AUC':>10}")
    print("-"*70)
    
    for name in results:
        rank = results[name]['ranking']
        cls = results[name]['classification']
        print(f"{name:<25} {rank['mrr']:>10.4f} {rank['hits@10']:>10.4f} "
              f"{cls['auc']:>10.4f}")
    
    print("="*70 + "\n")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Ejecutar configuración específica
        config_name = sys.argv[1]
        
        if config_name == 'grid_search':
            print("Generando configuraciones para grid search...")
            configs = generate_grid_search_configs()
            print(f"Total de configuraciones: {len(configs)}")
            print("\nPara ejecutar grid search, itera sobre las configs:")
            print("for config in configs:")
            print("    run_with_config(custom_config=config)")
            
        elif config_name == 'comparative':
            run_comparative_study()
            
        elif config_name == 'ablation':
            run_ablation_study()
            
        elif config_name == 'list':
            print("\nConfiguraciones disponibles:")
            print("-"*70)
            for name, config in CONFIGS.items():
                print(f"\n{name}:")
                print(f"  {config['description']}")
                print(f"  Dataset: {config['dataset_name']}, Modo: {config['mode']}")
            
        else:
            # Ejecutar configuración por nombre
            run_with_config(config_name)
    
    else:
        # Por defecto: mostrar ayuda
        print("\n" + "="*70)
        print("EJEMPLOS DE USO")
        print("="*70)
        print("\n1. Ejecutar configuración específica:")
        print("   python run_experiments.py codex_standard")
        print("   python run_experiments.py wordnet")
        print("   python run_experiments.py codex_ookb")
        
        print("\n2. Listar configuraciones disponibles:")
        print("   python run_experiments.py list")
        
        print("\n3. Ejecutar estudio comparativo:")
        print("   python run_experiments.py comparative")
        
        print("\n4. Ejecutar estudio de ablation:")
        print("   python run_experiments.py ablation")
        
        print("\n5. Generar configuraciones para grid search:")
        print("   python run_experiments.py grid_search")
        
        print("\n" + "="*70 + "\n")
