import warnings
from typing import Literal
import polars as pl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from umap import UMAP
from sknetwork.clustering import Leiden
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


get_x = lambda x: np.nan if not isinstance(x, (list, tuple, np.ndarray)) or len(x) < 1 else x[0]
get_y = lambda x: np.nan if not isinstance(x, (list, tuple, np.ndarray)) or len(x) < 2 else x[1]

def cosine_distance(X, Y):
    if isinstance(X, list): X = np.array(X)
    if isinstance(Y, list): Y = np.array(Y)
    return 1 - cosine_similarity(X.reshape(1, -1), Y.reshape(1, -1))[0][0]

def l2_distance(X, Y):
    if isinstance(X, list): X = np.array(X)
    if isinstance(Y, list): Y = np.array(Y)
    return np.linalg.norm(X - Y)

def dot_product(X, Y):
    # Dot product is a similarity measure, so we return 1 - dot product to convert it to a distance
    if isinstance(X, list): X = np.array(X)
    if isinstance(Y, list): Y = np.array(Y)
    return 1 - np.dot(X, Y)

def dispersion(centroid, embeddings, distance_func):
    """
    Calculate dispersion of a centroid from a set of embeddings.
    """
    distances = [distance_func(centroid, embedding) for embedding in embeddings]
    return np.mean(distances)

class TopolModeling:
    def __init__(self, n_components, umap_model_params, leiden_model_params, random_state=42, ignore_warnings=True):
        if ignore_warnings:
            warnings.filterwarnings("ignore")
        if "transform_mode" in umap_model_params and umap_model_params["transform_mode"] != "graph":
            warnings.warn("UMAP transform_mode will be overridden to 'graph' for Leiden clustering (and 'embedding' for visualization).")
        self.umap_graph = UMAP(n_components=n_components, transform_mode="graph", **umap_model_params, random_state=random_state)
        self.umap_dim_reducer = UMAP(n_components=n_components, transform_mode="embedding", **umap_model_params, random_state=random_state)
        self.umap_2D_model = UMAP(n_components=2, transform_mode="embedding", **umap_model_params, random_state=random_state)
        self.leiden_model = Leiden(**leiden_model_params, random_state=random_state)
        self.random_state = random_state
        self.df = None

    def _apply_umap(self):
        # embeddings = np.array(self.df["embedding"].to_list())
        embeddings = np.stack(self.df["embedding"].to_numpy()) # Faster
        self.umap_graph.fit(embeddings)
        reduced_embeddings = self.umap_dim_reducer.fit_transform(embeddings)
        reduced_2D_embeddings = self.umap_2D_model.fit_transform(embeddings)

        self.df = self.df.with_columns(
            pl.Series("reduced_embedding", reduced_embeddings, dtype=pl.List(pl.Float64)),
            pl.Series("2D_embedding", reduced_2D_embeddings, dtype=pl.List(pl.Float64))
        )
        print("UMAP applied successfully.")

    def _apply_leiden(self):
        adjacency_matrix = csr_matrix(self.umap_graph.graph_)

        cluster_labels = self.leiden_model.fit_predict(adjacency_matrix)
        cluster_all_probs = self.leiden_model.predict_proba()
        cluster_probs = [probs[cluster_id] for probs, cluster_id in zip(cluster_all_probs, cluster_labels)]

        self.df = self.df.with_columns(
            pl.Series("cluster", cluster_labels, dtype=pl.Int32),
            pl.Series("cluster_prob", cluster_probs, dtype=pl.Float64),
            # pl.Series("cluster_all_prob", cluster_all_probs, dtype=pl.List(pl.Float64))
        )
        print("Leiden clustering applied successfully.")

    def fit(self, df: pl.DataFrame):
        if "label" not in df.columns:
            raise ValueError("Dataframe must contain a 'label' column for fitting.")
        if "embedding" not in df.columns:
            raise ValueError("Dataframe must contain an 'embedding' column for fitting.")
        self.df = df.clone()
        self.df = self.df.with_columns(
            pl.Series("random_label", self.df["label"].sample(fraction=1, shuffle=True, seed=self.random_state).cast(pl.Float64)),
        )
        self._apply_umap()
        self._apply_leiden()

    def statistical_test(self, n_simulations: int = 1000) -> float:
        pass

    def get_cluster_info(self, label_col: Literal["label", "random_label"] = "label"):
        prefix = "" # "" or "reduced_"
        contextual_boundary_0 = self.df.filter(pl.col(label_col) == 0)
        cluster_info_0 = (
            contextual_boundary_0.group_by("cluster")
            .map_groups(lambda group: pl.DataFrame({
                "cluster": group["cluster"][0],
                "size": group.height,
                "avg_prob": [np.mean(group["cluster_prob"].to_list())],
                "centroid": [np.mean(group["embedding"].to_list(), axis=0)],
                "reduced_centroid": [np.mean(group["reduced_embedding"].to_list(), axis=0)],
                "2D_centroid": [np.mean(group["2D_embedding"].to_list(), axis=0)],
            }))
        )
        cluster_info_0 = cluster_info_0.with_columns([
            pl.struct(["cluster", prefix+"centroid"]).map_elements(lambda row: dispersion(row[prefix+"centroid"], contextual_boundary_0.filter(pl.col("cluster") == row["cluster"])[prefix+"embedding"].to_list(), cosine_distance), return_dtype=pl.Float64).alias("cosine_dispersion"),
            pl.struct(["cluster", prefix+"centroid"]).map_elements(lambda row: dispersion(row[prefix+"centroid"], contextual_boundary_0.filter(pl.col("cluster") == row["cluster"])[prefix+"embedding"].to_list(), l2_distance), return_dtype=pl.Float64).alias("l2_dispersion"),
            pl.struct(["cluster", prefix+"centroid"]).map_elements(lambda row: dispersion(row[prefix+"centroid"], contextual_boundary_0.filter(pl.col("cluster") == row["cluster"])[prefix+"embedding"].to_list(), dot_product), return_dtype=pl.Float64).alias("dot_product_dispersion"),
        ])
        
        contextual_boundary_1 = self.df.filter(pl.col(label_col) == 1)
        cluster_info_1 = (
            contextual_boundary_1.group_by("cluster")
            .map_groups(lambda group: pl.DataFrame({
                "cluster": group["cluster"][0],
                "size": group.height,
                "avg_prob": [np.mean(group["cluster_prob"].to_list())],
                "centroid": [np.mean(group["embedding"].to_list(), axis=0)],
                "reduced_centroid": [np.mean(group["reduced_embedding"].to_list(), axis=0)],
                "2D_centroid": [np.mean(group["2D_embedding"].to_list(), axis=0)],
            }))
        )
        cluster_info_1 = cluster_info_1.with_columns([
            pl.struct(["cluster", prefix+"centroid"]).map_elements(lambda row: dispersion(row[prefix+"centroid"], contextual_boundary_1.filter(pl.col("cluster") == row["cluster"])[prefix+"embedding"].to_list(), cosine_distance), return_dtype=pl.Float64).alias("cosine_dispersion"),
            pl.struct(["cluster", prefix+"centroid"]).map_elements(lambda row: dispersion(row[prefix+"centroid"], contextual_boundary_1.filter(pl.col("cluster") == row["cluster"])[prefix+"embedding"].to_list(), l2_distance), return_dtype=pl.Float64).alias("l2_dispersion"),
            pl.struct(["cluster", prefix+"centroid"]).map_elements(lambda row: dispersion(row[prefix+"centroid"], contextual_boundary_1.filter(pl.col("cluster") == row["cluster"])[prefix+"embedding"].to_list(), dot_product), return_dtype=pl.Float64).alias("dot_product_dispersion"),
        ])

        return cluster_info_0.sort("cluster"), cluster_info_1.sort("cluster")

    def visualize(self, label_col: Literal["label", "random_label"] = "label", figsize = (10, 10)):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
         
        # --- Prepare data ---
        clusters = np.unique(self.df["cluster"].to_numpy())
        nb_clusters = len(clusters)
        a_color = mpl.colormaps['Blues'].resampled(nb_clusters)
        b_color = mpl.colormaps['Reds'].resampled(nb_clusters)
        norm_A = mcolors.BoundaryNorm(boundaries=np.arange(nb_clusters + 1) - 0.5, ncolors=nb_clusters)
        norm_B = mcolors.BoundaryNorm(boundaries=np.arange(nb_clusters + 1) - 0.5, ncolors=nb_clusters)

        # --- Plot document embeddings ---
        # data_A = self.df[self.df[label_col] == 0] = contextual boundary with label 0
        data_A = self.df.filter(pl.col(label_col) == 0)
        scatter_A = ax.scatter(
            [get_x(e) for e in data_A["2D_embedding"].to_list()],
            [get_y(e) for e in data_A["2D_embedding"].to_list()],
            c=data_A["cluster"].to_list(),
            cmap=a_color,
            s=20,
            label="Dataset A - Doc. Embeddings",
            alpha=0.7
        )

        # data_B = self.df[self.df[label_col] == 1] = contextual boundary with label 1
        data_B = self.df.filter(pl.col(label_col) == 1)
        scatter_B = ax.scatter(
            [get_x(e) for e in data_B["2D_embedding"].to_list()],
            [get_y(e) for e in data_B["2D_embedding"].to_list()],
            c=data_B["cluster"].to_list(),
            cmap=b_color,
            s=20,
            label="Dataset B - Doc. Embeddings",
            alpha=0.7
        )

        # --- Plot drift arrows ---
        cluster_info_0, cluster_info_1 = self.get_cluster_info(label_col=label_col)
        drifts = {}
        for row in cluster_info_1.iter_rows(named=True):
            cluster_id = row['cluster']
            if cluster_id in cluster_info_0['cluster'].to_numpy():
                centroid_A_2D = cluster_info_0.filter(pl.col("cluster") == cluster_id)["2D_centroid"].to_numpy()[0]
                centroid_B_2D = row["2D_centroid"]
                start_x, start_y = get_x(centroid_A_2D), get_y(centroid_A_2D)
                end_x, end_y = get_x(centroid_B_2D), get_y(centroid_B_2D)
                ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y), arrowprops=dict(arrowstyle="->", color="black", lw=2))
                drifts[cluster_id] = np.array(centroid_B_2D) - np.array(centroid_A_2D)
            else:
                drifts[cluster_id] = np.nan

        # --- Add text labels for clusters ---
        for row in cluster_info_0.iter_rows(named=True):
            cluster_id = row['cluster']
            centroid_A_2D = row["2D_centroid"]
            centroid_B_2D = cluster_info_1.filter(pl.col("cluster") == cluster_id)["2D_centroid"].to_numpy()[0]
            start_x, start_y = get_x(centroid_A_2D), get_y(centroid_A_2D)
            end_x, end_y = get_x(centroid_B_2D), get_y(centroid_B_2D)
            ax.text(
                start_x, start_y, # Start point
                f"A{cluster_id}", fontsize=6, ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
            )
            ax.text(
                end_x, end_y, # End point
                f"B{cluster_id}", fontsize=6, ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
            )

        # --- Add color bar ---
        cbar_A = fig.colorbar(
            ScalarMappable(norm=norm_A, cmap=a_color), ax=ax, orientation="vertical", location="left", pad=0.15, ticks=clusters
        ); cbar_A.set_label("Cluster ID (A)", fontsize=10)
        cbar_B = fig.colorbar(
            ScalarMappable(norm=norm_B, cmap=b_color), ax=ax, orientation="vertical", location="right", pad=0.15, ticks=clusters
        ); cbar_B.set_label("Cluster ID (B)", fontsize=10)

        # --- Final styling ---
        legend_dot_A = mlines.Line2D([], [], color='#3787c0', marker='o', linestyle='None', markersize=6, label='Period A - Doc. Embeddings')
        legend_dot_B = mlines.Line2D([], [], color='#e32f27', marker='o', linestyle='None', markersize=6, label='Period B - Doc. Embeddings')
        ax.legend(handles=[legend_dot_A, legend_dot_B], loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
        print("Drift computed successfully, ready to visualize.")
        return fig, ax