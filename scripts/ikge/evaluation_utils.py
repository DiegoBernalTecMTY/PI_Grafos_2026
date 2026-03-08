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
        self.group_data = None
        self.model_name = "Modelo Desconocido"

    def evaluate_ranking(self, predict_fn, test_triples, num_entities,
                         batch_size=128, k_values=[1, 3, 10],
                         higher_is_better=True, verbose=True,
                         filter_tails=None, filter_heads=None):
        """
        Filtered bidirectional MRR (head+tail corruption).
        Used for validation during training to give the scheduler a scalar signal.
        """
        dev = torch.device(self.device)
        ranks_tail = []
        ranks_head = []

        test_t  = torch.tensor(test_triples, dtype=torch.long, device=dev)
        n_test  = test_t.size(0)
        ent_rng = torch.arange(num_entities, device=dev)
        NEG_INF = float('-inf')

        if verbose:
            print(f"--- Validation: filtered ranking on {n_test} triples (head+tail) ---")

        with torch.no_grad():
            for i in tqdm(range(0, n_test, batch_size), disable=not verbose):
                batch = test_t[i:i + batch_size]
                B     = len(batch)
                heads = batch[:, 0]; rels = batch[:, 1]; tails = batch[:, 2]

                # Tail prediction
                exp_h = heads.unsqueeze(1).expand(B, num_entities).reshape(-1)
                exp_r = rels.unsqueeze(1).expand(B, num_entities).reshape(-1)
                exp_t = ent_rng.unsqueeze(0).expand(B, -1).reshape(-1)
                ts = predict_fn(exp_h, exp_r, exp_t).float().view(B, num_entities)
                if filter_tails:
                    for j in range(B):
                        known = filter_tails.get((heads[j].item(), rels[j].item()))
                        if known:
                            for e in known:
                                if e != tails[j].item(): ts[j, e] = NEG_INF
                pos = ts[torch.arange(B, device=dev), tails]
                ranks_tail.extend(((ts >= pos.unsqueeze(1)).sum(1)).cpu().tolist())

                # Head prediction
                exp_h2 = ent_rng.unsqueeze(0).expand(B, -1).reshape(-1)
                exp_r2 = rels.unsqueeze(1).expand(B, num_entities).reshape(-1)
                exp_t2 = tails.unsqueeze(1).expand(B, num_entities).reshape(-1)
                hs = predict_fn(exp_h2, exp_r2, exp_t2).float().view(B, num_entities)
                if filter_heads:
                    for j in range(B):
                        known = filter_heads.get((rels[j].item(), tails[j].item()))
                        if known:
                            for e in known:
                                if e != heads[j].item(): hs[j, e] = NEG_INF
                pos = hs[torch.arange(B, device=dev), heads]
                ranks_head.extend(((hs >= pos.unsqueeze(1)).sum(1)).cpu().tolist())

        ranks = np.array(ranks_tail + ranks_head)
        metrics = {
            'mrr':      float(np.mean(1.0 / ranks)),
            'mr':       float(np.mean(ranks)),
            'mrr_tail': float(np.mean(1.0 / np.array(ranks_tail))),
            'mrr_head': float(np.mean(1.0 / np.array(ranks_head))),
        }
        for k in k_values:
            metrics[f'hits@{k}'] = float(np.mean(ranks <= k))

        self.ranking_data = {'ranks': ranks, 'metrics': metrics, 'k_values': k_values}

        if verbose:
            print(f"  Val MRR (filtered): {metrics['mrr']:.4f}  "
                  f"| tail {metrics['mrr_tail']:.4f}  head {metrics['mrr_head']:.4f}")
            for k in k_values:
                print(f"  Hits@{k}: {metrics[f'hits@{k}']:.4f}")
        return metrics

    # ------------------------------------------------------------------
    # Paper-exact group evaluation (test set only)
    # ------------------------------------------------------------------

    def _rank_one_direction(self, predict_fn, triples, num_cands,
                             corrupt_dim, batch_size, k_values, filter_dict, desc):
        """
        Filtered ranking for a single corruption direction.

        corrupt_dim : 'head' | 'tail' | 'relation'
        filter_dict :
            'tail'     -> {(h,r): [t1,t2,...]}  (known correct tails)
            'head'     -> {(r,t): [h1,h2,...]}  (known correct heads)
            'relation' -> {(h,t): [r1,r2,...]}  (known correct relations)
        Returns (list_of_ranks, n_triples)
        """
        if not triples:
            return [], 0

        dev     = torch.device(self.device)
        NEG_INF = float('-inf')
        ranks   = []

        test_t   = torch.tensor(triples, dtype=torch.long, device=dev)
        n        = test_t.size(0)
        cand_rng = torch.arange(num_cands, device=dev)

        with torch.no_grad():
            for i in tqdm(range(0, n, batch_size), desc=desc):
                batch  = test_t[i:i + batch_size]
                B      = len(batch)
                heads  = batch[:, 0]
                rels   = batch[:, 1]
                tails  = batch[:, 2]

                if corrupt_dim == 'tail':
                    exp_h = heads.unsqueeze(1).expand(B, num_cands).reshape(-1)
                    exp_r = rels.unsqueeze(1).expand(B, num_cands).reshape(-1)
                    exp_c = cand_rng.unsqueeze(0).expand(B, -1).reshape(-1)
                    scores  = predict_fn(exp_h, exp_r, exp_c).float().view(B, num_cands)
                    targets = tails
                    if filter_dict:
                        for j in range(B):
                            known = filter_dict.get((heads[j].item(), rels[j].item()))
                            if known:
                                for e in known:
                                    if e != tails[j].item(): scores[j, e] = NEG_INF

                elif corrupt_dim == 'head':
                    exp_c = cand_rng.unsqueeze(0).expand(B, -1).reshape(-1)
                    exp_r = rels.unsqueeze(1).expand(B, num_cands).reshape(-1)
                    exp_t = tails.unsqueeze(1).expand(B, num_cands).reshape(-1)
                    scores  = predict_fn(exp_c, exp_r, exp_t).float().view(B, num_cands)
                    targets = heads
                    if filter_dict:
                        for j in range(B):
                            known = filter_dict.get((rels[j].item(), tails[j].item()))
                            if known:
                                for e in known:
                                    if e != heads[j].item(): scores[j, e] = NEG_INF

                else:  # 'relation'
                    exp_h = heads.unsqueeze(1).expand(B, num_cands).reshape(-1)
                    exp_c = cand_rng.unsqueeze(0).expand(B, -1).reshape(-1)
                    exp_t = tails.unsqueeze(1).expand(B, num_cands).reshape(-1)
                    scores  = predict_fn(exp_h, exp_c, exp_t).float().view(B, num_cands)
                    targets = rels
                    if filter_dict:
                        for j in range(B):
                            known = filter_dict.get((heads[j].item(), tails[j].item()))
                            if known:
                                for e in known:
                                    if e != rels[j].item(): scores[j, e] = NEG_INF

                pos = scores[torch.arange(B, device=dev), targets]
                ranks.extend(((scores >= pos.unsqueeze(1)).sum(1)).cpu().tolist())

        return ranks, n

    def evaluate_ikge_groups(self, predict_fn,
                              group1_triples, group2_triples,
                              group3_oot_triples, group3_xoo_triples,
                              group4_triples,
                              num_entities, num_relations,
                              batch_size=256, k_values=[1, 3, 10],
                              filter_tails=None, filter_heads=None, filter_rels=None):
        """
        Evaluates the 4 IKGE paper groups exactly as defined in Table 2.

        Group 1 (head entity pred, paper MRR=0.34):
            Predict h on: O-O-X + O-X-O + O-X-X
        Group 2 (tail entity pred, paper MRR=0.61):
            Predict t on: X-O-O + O-X-O + X-X-O
        Group 3 (head+tail entity pred, paper MRR=0.52):
            Predict t on O-O-X (tail is out-of-KG)
            Predict h on X-O-O (head is out-of-KG)
        Group 4 (relation pred, paper MRR=0.31):
            Predict r on: O-O-X + X-O-O

        Returns dict of per-group metrics + 'overall' key.
        """
        groups = {
            'Group 1 - Head entity prediction': [
                ('head', group1_triples, num_entities, filter_heads)
            ],
            'Group 2 - Tail entity prediction': [
                ('tail', group2_triples, num_entities, filter_tails)
            ],
            'Group 3 - Head+Tail entity prediction': [
                # O-O-X: tail is OOK → paper says HEAD entity prediction (predict in-KG head)
                ('head', group3_oot_triples, num_entities, filter_heads),
                # X-O-O: head is OOK → paper says TAIL entity prediction (predict in-KG tail)
                ('tail', group3_xoo_triples, num_entities, filter_tails)
            ],
            'Group 4 - Relation prediction': [
                ('relation', group4_triples, num_relations, filter_rels)
            ],
        }

        all_group_ranks = []
        results = {}

        for group_name, directions in groups.items():
            group_ranks = []
            total = 0
            for (corrupt_dim, triples, num_cands, fdict) in directions:
                desc  = f"{group_name} [{corrupt_dim}]"
                r, n  = self._rank_one_direction(
                    predict_fn, triples, num_cands,
                    corrupt_dim, batch_size, k_values, fdict, desc
                )
                group_ranks.extend(r)
                total += n

            if not group_ranks:
                results[group_name] = {'mrr': 0.0, 'mr': 0.0, 'n': 0}
                for k in k_values:
                    results[group_name][f'hits@{k}'] = 0.0
                continue

            arr = np.array(group_ranks)
            gm  = {'mrr': float(np.mean(1.0 / arr)),
                   'mr':  float(np.mean(arr)),
                   'n':   total}
            for k in k_values:
                gm[f'hits@{k}'] = float(np.mean(arr <= k))
            results[group_name] = gm
            all_group_ranks.extend(group_ranks)

        # Overall metrics across all evaluated triples
        if all_group_ranks:
            arr = np.array(all_group_ranks)
            results['overall'] = {
                'mrr':   float(np.mean(1.0 / arr)),
                'mr':    float(np.mean(arr)),
                'n':     len(arr),
            }
            for k in k_values:
                results['overall'][f'hits@{k}'] = float(np.mean(arr <= k))

        # Pretty print
        sep = '─' * 70
        print(f"\n{sep}")
        print(f"  IKGE Paper-Exact Group Evaluation")
        print(sep)
        hdr = f"  {'Group':<42} {'n':>5}  {'MRR':>7}  {'H@1':>7}  {'H@3':>7}  {'H@10':>7}"
        print(hdr)
        print(sep)
        paper_mrr = {
            'Group 1 - Head entity prediction':      0.34,
            'Group 2 - Tail entity prediction':      0.61,
            'Group 3 - Head+Tail entity prediction': 0.52,
            'Group 4 - Relation prediction':         0.31,
        }
        for gname, gm in results.items():
            if gname == 'overall':
                continue
            paper = paper_mrr.get(gname, 0)
            diff  = gm['mrr'] - paper
            flag  = f" (paper {paper:.2f}, diff {diff:+.2f})"
            print(f"  {gname:<42} {gm['n']:>5}  {gm['mrr']:>7.4f}  "
                  f"{gm.get('hits@1',0):>7.4f}  {gm.get('hits@3',0):>7.4f}  "
                  f"{gm.get('hits@10',0):>7.4f}{flag}")
        if 'overall' in results:
            ov = results['overall']
            print(sep)
            print(f"  {'Overall (all groups)':<42} {ov['n']:>5}  {ov['mrr']:>7.4f}  "
                  f"{ov.get('hits@1',0):>7.4f}  {ov.get('hits@3',0):>7.4f}  "
                  f"{ov.get('hits@10',0):>7.4f}")
        print(f"{sep}\n")

        # Store for report
        self.group_data = results
        if all_group_ranks:
            self.ranking_data = {
                'ranks':   np.array(all_group_ranks),
                'metrics': results.get('overall', {}),
                'k_values': k_values
            }

        return results

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
            if self.group_data:
                lines = ["IKGE Paper-Exact Group Evaluation:\n",
                         "------------------------------------------\n"]
                for gname, gm in self.group_data.items():
                    if gname == 'overall':
                        continue
                    lines.append(f"{gname[:40]:<40}  n={gm.get('n',0):>5}  "
                                 f"MRR={gm.get('mrr',0):.4f}  "
                                 f"H@1={gm.get('hits@1',0):.4f}  "
                                 f"H@10={gm.get('hits@10',0):.4f}\n")
                if 'overall' in self.group_data:
                    ov = self.group_data['overall']
                    lines.append(f"{'Overall':<40}  n={ov.get('n',0):>5}  "
                                 f"MRR={ov.get('mrr',0):.4f}  "
                                 f"H@1={ov.get('hits@1',0):.4f}  "
                                 f"H@10={ov.get('hits@10',0):.4f}\n")
                plt.text(0.05, 0.50, ''.join(lines), fontsize=9,
                         family='monospace', va='top')
            elif self.ranking_data:
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