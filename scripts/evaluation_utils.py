import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, auc, 
                             accuracy_score, f1_score, confusion_matrix, 
                             classification_report)
from tqdm import tqdm
import pandas as pd

class UnifiedKGScorer:
    """
    Clase estandarizada para evaluar modelos de Knowledge Graph Completion.
    Genera reportes en PDF con gráficas y métricas en español.
    """
    def __init__(self, device='cuda'):
        self.device = device
        # Almacenamiento interno para el reporte
        self.ranking_data = None
        self.class_data = None
        self.model_name = "Modelo Desconocido"

    def evaluate_ranking(self, predict_fn, test_triples, num_entities, 
                         batch_size=128, k_values=[1, 3, 10], 
                         higher_is_better=True, verbose=True):
        """Evalúa métricas de Ranking (MRR, Hits@K)."""
        ranks = []
        test_triples = torch.tensor(test_triples, device=self.device)
        n_test = test_triples.size(0)

        if verbose:
            print(f"--- Evaluando Ranking en {n_test} tripletas ---")

        # Modo evaluación para ahorrar memoria
        with torch.no_grad():
            for i in tqdm(range(0, n_test, batch_size), disable=not verbose):
                batch = test_triples[i:i+batch_size]
                heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]

                # Score Target
                pos_scores = predict_fn(heads, rels, tails)

                # Corrupción de Colas (Batch optimizado)
                # Evaluamos contra todas las entidades
                batch_heads = heads.unsqueeze(1).repeat(1, num_entities).view(-1)
                batch_rels  = rels.unsqueeze(1).repeat(1, num_entities).view(-1)
                batch_tails = torch.arange(num_entities, device=self.device).repeat(len(batch))

                all_scores = predict_fn(batch_heads, batch_rels, batch_tails)
                all_scores = all_scores.view(len(batch), num_entities)

                # Calcular rangos
                for j in range(len(batch)):
                    target_score = pos_scores[j].item()
                    row_scores = all_scores[j]

                    if higher_is_better:
                        better_count = (row_scores > target_score).sum().item()
                    else:
                        better_count = (row_scores < target_score).sum().item()
                    
                    ranks.append(better_count + 1)

        ranks = np.array(ranks)
        metrics = {
            'mrr': np.mean(1.0 / ranks),
            'mr': np.mean(ranks),
        }
        for k in k_values:
            metrics[f'hits@{k}'] = np.mean(ranks <= k)

        # Guardar para el reporte
        self.ranking_data = {
            'ranks': ranks,
            'metrics': metrics,
            'k_values': k_values
        }
        
        if verbose:
            print(f"Resultados Ranking: {metrics}")
            
        return metrics

    def evaluate_classification(self, predict_fn, valid_pos, test_pos, 
                                num_entities, higher_is_better=True):
        """Evalúa Triple Classification y guarda datos para curvas ROC/PR."""
        print("--- Evaluando Triple Classification ---")
        
        # Generar Negativos
        valid_neg = self._generate_negatives(valid_pos, num_entities)
        test_neg = self._generate_negatives(test_pos, num_entities)

        # Scores
        val_pos_scores = self._batch_predict(predict_fn, valid_pos)
        val_neg_scores = self._batch_predict(predict_fn, valid_neg)
        test_pos_scores = self._batch_predict(predict_fn, test_pos)
        test_neg_scores = self._batch_predict(predict_fn, test_neg)

        # Etiquetas (1=Positivo, 0=Negativo)
        y_val = np.concatenate([np.ones(len(val_pos_scores)), np.zeros(len(val_neg_scores))])
        y_test = np.concatenate([np.ones(len(test_pos_scores)), np.zeros(len(test_neg_scores))])
        
        scores_val = np.concatenate([val_pos_scores, val_neg_scores])
        scores_test = np.concatenate([test_pos_scores, test_neg_scores])

        # Normalizar scores para AUC si es métrica de distancia
        if not higher_is_better:
            scores_val = -scores_val
            scores_test = -scores_test

        # Encontrar el mejor Umbral en Validación
        best_acc = 0
        best_thresh = 0
        thresholds = np.unique(np.percentile(scores_val, np.arange(0, 100, 1)))
        
        for t in thresholds:
            preds = (scores_val >= t).astype(int)
            acc = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = t

        print(f"  Umbral óptimo (Validación): {best_thresh:.4f}")

        # Predicciones finales en Test
        final_preds = (scores_test >= best_thresh).astype(int)
        
        # Métricas detalladas
        metrics = {
            'auc': 0.0, # Se calcula abajo
            'accuracy': accuracy_score(y_test, final_preds),
            'f1': f1_score(y_test, final_preds),
            'confusion_matrix': confusion_matrix(y_test, final_preds)
        }
        
        # Calcular curvas para reporte
        fpr, tpr, _ = roc_curve(y_test, scores_test)
        roc_auc = auc(fpr, tpr)
        metrics['auc'] = roc_auc
        
        precision, recall, _ = precision_recall_curve(y_test, scores_test)

        # Guardar para el reporte
        self.class_data = {
            'y_true': y_test,
            'y_scores': scores_test,
            'y_pred': final_preds,
            'pos_scores': test_pos_scores if higher_is_better else -test_pos_scores,
            'neg_scores': test_neg_scores if higher_is_better else -test_neg_scores,
            'threshold': best_thresh,
            'metrics': metrics,
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
            'prec_curve': precision, 'rec_curve': recall
        }

        return metrics

    def export_report(self, model_name, filename="reporte_modelo.pdf"):
        """
        Genera un PDF completo en español con gráficas y tablas.
        """
        print(f"--- Generando reporte PDF: {filename} ---")
        self.model_name = model_name
        
        with PdfPages(filename) as pdf:
            # --- PÁGINA 1: Resumen Ejecutivo ---
            plt.figure(figsize=(10, 12))
            plt.axis('off')
            
            # Título
            plt.text(0.5, 0.95, f"Reporte de Evaluación de Modelo\n{self.model_name}", 
                     ha='center', va='center', fontsize=20, weight='bold')
            
            # Tabla de Métricas de Clasificación
            if self.class_data:
                m = self.class_data['metrics']
                text_class = (
                    f"Métricas de Clasificación (Triple Classification):\n"
                    f"--------------------------------------------\n"
                    f"Área bajo la curva (AUC): {m['auc']:.4f}\n"
                    f"Exactitud (Accuracy):     {m['accuracy']:.4f}\n"
                    f"F1-Score:                 {m['f1']:.4f}\n"
                    f"Umbral Óptimo:            {self.class_data['threshold']:.4f}\n"
                )
                plt.text(0.1, 0.75, text_class, fontsize=12, family='monospace')

            # Tabla de Métricas de Ranking
            if self.ranking_data:
                r = self.ranking_data['metrics']
                text_rank = (
                    f"Métricas de Ranking (Link Prediction):\n"
                    f"--------------------------------------------\n"
                    f"MRR (Mean Reciprocal Rank): {r['mrr']:.4f}\n"
                    f"MR (Mean Rank):             {r['mr']:.2f}\n"
                    f"Hits@1:                     {r.get('hits@1', 0):.4f}\n"
                    f"Hits@3:                     {r.get('hits@3', 0):.4f}\n"
                    f"Hits@10:                    {r.get('hits@10', 0):.4f}\n"
                )
                plt.text(0.1, 0.50, text_rank, fontsize=12, family='monospace')
            
            plt.text(0.5, 0.1, "Generado automáticamente por UnifiedKGScorer", 
                     ha='center', fontsize=8, color='gray')
            pdf.savefig()
            plt.close()

            # --- PÁGINA 2: Curvas de Rendimiento (ROC y PR) ---
            if self.class_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # ROC Curve
                ax1.plot(self.class_data['fpr'], self.class_data['tpr'], 
                         color='darkorange', lw=2, label=f'AUC = {self.class_data["roc_auc"]:.2f}')
                ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax1.set_xlabel('Tasa de Falsos Positivos')
                ax1.set_ylabel('Tasa de Verdaderos Positivos')
                ax1.set_title('Curva ROC')
                ax1.legend(loc="lower right")
                ax1.grid(True, alpha=0.3)

                # Precision-Recall
                ax2.plot(self.class_data['rec_curve'], self.class_data['prec_curve'], 
                         color='green', lw=2)
                ax2.set_xlabel('Sensibilidad (Recall)')
                ax2.set_ylabel('Precisión')
                ax2.set_title('Curva Precisión-Recall')
                ax2.grid(True, alpha=0.3)
                
                plt.suptitle(f"Análisis de Clasificación - {self.model_name}")
                pdf.savefig()
                plt.close()

                # --- PÁGINA 3: Separabilidad de Clases ---
                plt.figure(figsize=(10, 6))
                sns.kdeplot(self.class_data['pos_scores'], fill=True, color='green', label='Hechos Reales (Positivos)')
                sns.kdeplot(self.class_data['neg_scores'], fill=True, color='red', label='Hechos Falsos (Negativos)')
                plt.axvline(self.class_data['threshold'], color='black', linestyle='--', label='Umbral de Decisión')
                plt.title("Distribución de Puntuaciones (Scores)")
                plt.xlabel("Score del Modelo (Mayor es mejor)")
                plt.ylabel("Densidad")
                plt.legend()
                plt.grid(True, alpha=0.3)
                pdf.savefig()
                plt.close()

            # --- PÁGINA 4: Análisis de Ranking ---
            if self.ranking_data:
                plt.figure(figsize=(10, 6))
                ranks = self.ranking_data['ranks']
                # Histograma en escala logarítmica porque los rangos suelen ser extremos
                plt.hist(ranks, bins=30, color='purple', alpha=0.7, log=True)
                plt.title("Distribución de Rangos (Escala Logarítmica)")
                plt.xlabel("Rango Predicho (Menor es mejor)")
                plt.ylabel("Frecuencia (Log)")
                plt.grid(True, alpha=0.3)
                pdf.savefig()
                plt.close()

        print(f"Reporte guardado exitosamente en: {filename}")

    def _generate_negatives(self, triples, num_entities):
        """Generador interno de negativos."""
        negatives = triples.clone() if torch.is_tensor(triples) else torch.tensor(triples)
        negatives = negatives.to(self.device)
        mask = torch.rand(len(negatives), device=self.device) < 0.5
        rand_h = torch.randint(num_entities, (mask.sum(),), device=self.device)
        negatives[mask, 0] = rand_h
        rand_t = torch.randint(num_entities, ((~mask).sum(),), device=self.device)
        negatives[~mask, 2] = rand_t
        return negatives

    def _batch_predict(self, predict_fn, triples, batch_size=1024):
        """Helper para predicción por lotes."""
        triples = torch.tensor(triples, device=self.device)
        all_scores = []
        # Modo evaluación
        with torch.no_grad():
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i+batch_size]
                scores = predict_fn(batch[:, 0], batch[:, 1], batch[:, 2])
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)