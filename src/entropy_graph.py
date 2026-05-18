"""EntropyTransferGraph — turn DTW distances into an information-flow network.

Pipeline
--------
1. S = exp(-λ D)              similarity from DTW distances
2. p_i = S_i / Σ_j S_ij        row-normalised distribution per asset
3. H(i) = -Σ p_ij log p_ij      Shannon entropy per asset
4. M_ij = 1 - JSD(p_i, p_j)    Jensen-Shannon-based transfer matrix
5. Δ = 1 - M  →  embed via MDS or Spectral Embedding
6. Plot as a weighted network (networkx)

JSD is symmetric, bounded in [0, log 2] (with log base e), and a metric in
its square-root form. Ref: https://en.wikipedia.org/wiki/Jensen-Shannon_divergence
"""
import numpy as np
import pandas as pd


def _shannon(p: np.ndarray) -> float:
    # Classique H = - sum(p * log(p))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Distance de Jensen-Shannon. Parfait parce que c'est symétrique (contrairement à KL)."""
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


class EntropyTransferGraph:
    def __init__(self, distance_matrix: pd.DataFrame, lam: float = None):
        self.distance_matrix = distance_matrix
        self.lam = lam 
        
        self.similarity_ = None
        self.transfer_matrix_ = None
        self.entropy_ = None
        self.embedding_ = None

    def compute_embeddings(self, method: str = "mds", n_components: int = 2) -> dict:
        D = self.distance_matrix.values.astype(float)
        cols = list(self.distance_matrix.columns)

        # 1. Choix du paramètre lambda
        # L'idée c'est de forcer la médiane des similarités off-diagonal à 0.5
        if self.lam is None:
            offdiag = D[~np.eye(len(D), dtype=bool)]
            med = np.median(offdiag) if offdiag.size else 1.0
            self.lam = float(np.log(2) / med) if med > 0 else 1.0

        # 2. Similarité
        S = np.exp(-self.lam * D)
        np.fill_diagonal(S, 1.0)
        self.similarity_ = pd.DataFrame(S, index=cols, columns=cols)

        # 3. Distrib row-normalised (la "répartition" d'influence d'un noeud vs les autres)
        P = S / S.sum(axis=1, keepdims=True)

        # 4. Shannon (plus c'est bas, plus l'info est concentrée => "Leader potentiel")
        self.entropy_ = pd.Series([_shannon(P[i]) for i in range(len(cols))], index=cols, name="H")

        # 5. M_ij = 1 - JSD (plus JSD est faible, plus ils sont similaires)
        n = len(cols)
        M = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                jsd = _jsd(P[i], P[j])
                M[i, j] = M[j, i] = 1.0 - jsd
        self.transfer_matrix_ = pd.DataFrame(M, index=cols, columns=cols)

        # 6. Embedding 2D pour pouvoir dessiner le bordel
        Delta = 1.0 - M
        np.fill_diagonal(Delta, 0.0)
        Delta = np.clip(Delta, 0.0, None)

        if method == "mds":
            from sklearn.manifold import MDS
            emb = MDS(
                n_components=n_components,
                dissimilarity="precomputed",
                normalized_stress="auto",
                random_state=42 # random seed fixé pour la repoducibilité du prof
            ).fit_transform(Delta)
        elif method == "spectral":
            from sklearn.manifold import SpectralEmbedding
            emb = SpectralEmbedding(
                n_components=n_components,
                affinity="precomputed",
                random_state=42
            ).fit_transform(M)
        else:
            raise ValueError(f"Methode inconnue: {method}")

        self.embedding_ = pd.DataFrame(
            emb, index=cols, columns=[f"dim{i+1}" for i in range(n_components)]
        )
        return {
            "similarity": self.similarity_,
            "transfer": self.transfer_matrix_,
            "entropy": self.entropy_,
            "embedding": self.embedding_,
            "lambda": self.lam,
        }

    def plot_graph(self, threshold: float = 0.5, ax=None):
        """Dessine le network. Les nodes gros sont les "centralités" fortes."""
        import matplotlib.pyplot as plt
        import networkx as nx

        if self.transfer_matrix_ is None or self.embedding_ is None:
            self.compute_embeddings()

        M = self.transfer_matrix_
        G = nx.Graph()
        for c in M.columns:
            G.add_node(c)
            
        for i, a in enumerate(M.columns):
            for b in M.columns[i + 1 :]:
                w = float(M.loc[a, b])
                if w >= threshold: # coupe les edges ridicules
                    G.add_edge(a, b, weight=w)

        # Centralité pour la taille des points
        try:
            cent = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cent = dict(G.degree(weight="weight"))

        pos = {c: self.embedding_.loc[c, ["dim1", "dim2"]].values for c in M.columns}
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        sizes = [800 + 2000 * cent.get(c, 0) for c in G.nodes]
        weights = [G[u][v]["weight"] for u, v in G.edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4a90e2", alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=[3 * w for w in weights], alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title("Entropy Transfer Graph (MDS Projection)")
        ax.set_axis_off()
        return ax, G
