import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
import umap
from sknetwork.clustering import Leiden
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import block_diag, hstack, vstack
from sklearn.metrics.pairwise import cosine_similarity as cosine
from src.helpers.postprocessing import get_top_n_words, get_top_n_representative_documents, get_tf_idf_top_n_words


get_x = lambda x: np.nan if not isinstance(x, (list, tuple, np.ndarray)) or len(x) < 1 else x[0]
get_y = lambda x: np.nan if not isinstance(x, (list, tuple, np.ndarray)) or len(x) < 2 else x[1]

class Dataset(object):
    def __init__(self, df: pd.DataFrame):
        assert "text" in df.columns, "Dataframe must contain a 'text' column."
        assert "embedding" in df.columns, "Dataframe must contain an 'embeddings' column."
        self.df = pd.dfFrame(columns=["text", "embeddings"])
        self.df["reduced_embedding"] = None
        self.df["2D_embedding"] = None
        self.df["cluster"] = None
        self.df["cluster_prob"] = None
        self.cluster_info = None

class TopolModeling:
    def __init__(self, n_components, umap_model_params, leiden_model_params, vectorizer_model, supervised=False):
        self.umap_model = umap.UMAP(n_components=n_components, **umap_model_params)
        self.umap_2D_model = umap.UMAP(n_components=2, **umap_model_params)
        self.leiden_model = Leiden(**leiden_model_params)
        self.vectorizer_model = vectorizer_model
        self.supervised = supervised
        self.dataset_A = None
        self.dataset_B = None
        self.graph = None
        self.adjacency_matrix = None

    def _apply_umap(self):
        embeddings_A = np.stack(self.dataset_A.df["embedding"].values)
        embeddings_B = np.stack(self.dataset_B.df["embedding"].values)

        if self.supervised: # Known polarity separation
            embeddings_A_B = np.concatenate((embeddings_A, embeddings_B), axis=0)
            reduced_embeddings_A_B = self.umap_model.fit_transform(embeddings_A_B)
            reduced_2D_embeddings_A_B = self.umap_2D_model.fit_transform(embeddings_A_B)
            self.dataset_A.df["reduced_embedding"] = embeddings_A_B[:len(embeddings_A)]
            self.dataset_A.df["2D_embedding"] = reduced_2D_embeddings_A_B[:len(embeddings_A)]
            self.dataset_B.df["reduced_embedding"] = embeddings_A_B[len(embeddings_A):]
            self.dataset_B.df["2D_embedding"] = reduced_2D_embeddings_A_B[len(embeddings_A):]
        else: # Unsupervised polarity separation
            self.dataset_A.df["reduced_embedding"] = self.umap_model.fit_transform(embeddings_A)
            self.dataset_A.df["2D_embedding"] = self.umap_2D_model.fit_transform(embeddings_A)
            self.dataset_B.df["reduced_embedding"] = self.umap_model.transform(embeddings_B)
            self.dataset_B.df["2D_embedding"] = self.umap_2D_model.transform(embeddings_B)

    def _apply_leiden(self):
        if self.supervised: # Known polarity separation
            self.graph = self.umap_model.graph_
        else: # Unsupervised polarity separation
            embeddings_A = np.stack(self.dataset_A.df["reduced_embedding"].values)
            embeddings_B = np.stack(self.dataset_B.df["reduced_embedding"].values)
            graph_A = self.umap_model.graph_
            graph_B = umap.umap_.fuzzy_simplicial_set(
                X=embeddings_B,
                n_neighbors=self.umap_model.n_neighbors,
                random_state=self.umap_model.random_state,
                metric=self.umap_model.metric,
                metric_kwds=self.umap_model._metric_kwds,
                knn_indices=None,
                knn_dists=None,
                angular=self.umap_model.angular_rp_forest,
                set_op_mix_ratio=self.umap_model.set_op_mix_ratio,
                local_connectivity=self.umap_model.local_connectivity,
                verbose=False
            )[0]

            sim_matrix = cosine(embeddings_B, embeddings_A)
            # sim_matrix[sim_matrix < 0.75] = 0  

            n_B, n_A = sim_matrix.shape
            k = 10
            cross_edges = lil_matrix((n_B, n_A))
            for i in range(n_B):
                top_k_indices = np.argpartition(sim_matrix[i], -k)[-k:]
                cross_edges[i, top_k_indices] = sim_matrix[i, top_k_indices]
            cross_edges = csr_matrix(cross_edges)
            top = hstack([graph_A, cross_edges.T])
            bottom = hstack([cross_edges, graph_B])
            self.graph = vstack([top, bottom])

        self.adjacency_matrix = csr_matrix(self.graph)
        labels_A_B = self.leiden_model.fit_predict(self.adjacency_matrix)
        probs_A_B = self.leiden_model.predict_proba()
        self.dataset_A.df["cluster"] = labels_A_B[:len(self.dataset_A.df)]
        self.dataset_A.df["cluster_prob"] = probs_A_B[:len(self.dataset_A.df)]
        self.dataset_B.df["cluster"] = labels_A_B[len(self.dataset_A.df):]
        self.dataset_B.df["cluster_prob"] = probs_A_B[len(self.dataset_A.df):]


    def _create_cluster_info(self, df, n_top_freq_words, n_repr_docs):
        cluster_info = pd.DataFrame({"Cluster": np.unique(df['cluster'])})
        cluster_info["Count"] = cluster_info["Cluster"].map(df["cluster"].value_counts())
        cluster_info["Top_Words"] = cluster_info["Cluster"].apply(
            lambda x: get_top_n_words(df[df["cluster"] == x]["document"].values.tolist(), 
                                      self.vectorizer_model,
                                      n=n_top_freq_words)
        )
        # cluster_info["Polarity"] = cluster_info["Cluster"].apply(lambda x: df[df["cluster"] == x]["polarity"].mean())
        cluster_info["Centroid"] = cluster_info["Cluster"].apply(
            lambda x: np.mean(df[df["cluster"] == x]['embedding'].tolist(), axis=0)
        )
        cluster_info["2D_Centroid"] = cluster_info["Cluster"].apply(
            lambda x: np.mean(df[df["cluster"] == x]['2D_embeddings'].tolist(), axis=0)
        )
        cluster_info["Top_Representative_Docs"] = cluster_info[["Cluster", "Centroid"]].apply(
            lambda x:
                get_top_n_representative_documents(
                    df[df["cluster"] == x["Cluster"]]["document"].values.tolist(),
                    np.stack(df[df["cluster"] == x["Cluster"]]['embedding'].values), 
                    x["Centroid"],
                    n=n_repr_docs
                ),
                axis=1
        )
        return cluster_info

    def _exctract_info(self, n_top_freq_words, n_repr_docs, n_top_tf_idf_words):
        cluster_info_A = self._create_cluster_info(self.dataset_A.df, n_top_freq_words, n_repr_docs)
        cluster_info_A["Top_Words_TFIDF"] = [[] for _ in range(len(cluster_info_A))]
        cluster_info_B = self._create_cluster_info(self.dataset_B.df, n_top_freq_words, n_repr_docs)
        cluster_info_B["Top_Words_TFIDF"] = [[] for _ in range(len(cluster_info_B))]
        for cluster_id in cluster_info_A["Cluster"]:
            top_repr_doc_A = cluster_info_A[cluster_info_A["Cluster"] == cluster_id]["Top_Representative_Docs"].values[0]
            joined_top_repr_doc_A = " ".join(top_repr_doc_A)

            top_repr_doc_B = cluster_info_B[cluster_info_B["Cluster"] == cluster_id]["Top_Representative_Docs"].values[0]
            joined_top_repr_doc_B = " ".join(top_repr_doc_B)

            tf_idf_words_A_B = get_tf_idf_top_n_words([joined_top_repr_doc_A, joined_top_repr_doc_B], n=n_top_tf_idf_words)
            idx_A = cluster_info_A[cluster_info_A["Cluster"] == cluster_id].index[0]
            cluster_info_A.at[idx_A, "Top_Words_TFIDF"] = tf_idf_words_A_B[0]
            idx_B = cluster_info_B[cluster_info_B["Cluster"] == cluster_id].index[0]
            cluster_info_B.at[idx_B, "Top_Words_TFIDF"] = tf_idf_words_A_B[1]

    def vizualize_clusters(self, figsize=(10, 10)):
        fig, ax: plt.Axes = plt.subplots(figsize=figsize, layout="constrained")
         
        # --- Prepare data ---
        clusters_A = self.dataset_A.df["cluster"].unique()
        clusters_B = self.dataset_B.df["cluster"].unique()
        all_clusters = np.unique(np.concatenate((clusters_A, clusters_B)))
        total_clusters = len(all_clusters)
        a_color = mpl.colormaps['Blues'].resampled(total_clusters)
        b_color = mpl.colormaps['Reds'].resampled(total_clusters)
        norm_A = mcolors.BoundaryNorm(boundaries=np.arange(total_clusters + 1) - 0.5, ncolors=total_clusters)
        norm_B = mcolors.BoundaryNorm(boundaries=np.arange(total_clusters + 1) - 0.5, ncolors=total_clusters)

        # --- Plot document embeddings ---
        # Period A
        scatter_A = ax.scatter(
            self.dataset_A.df["2D_embeddings"].apply(get_x), self.dataset_A.df["2D_embeddings"].apply(get_y),
            c=self.dataset_A.df["cluster"],
            cmap=a_color,
            s=20,
            label="Dataset A - Doc. Embeddings",
            alpha=0.7
        )

        # Period B
        scatter_B = ax.scatter(
            self.dataset_B.df["2D_embeddings"].apply(get_x), self.dataset_B.df["2D_embeddings"].apply(get_y),
            c=self.dataset_B.df["cluster"],
            cmap=b_color,
            s=20,
            label="Dataset B - Doc. Embeddings",
            alpha=0.7
        )

        # --- Plot drift arrows ---
        drifts = {}
        for idx, row in self.cluster_info_B.iterrows():
            cluster_id = row['Cluster']
            if cluster_id in self.cluster_info_A['Cluster'].values:
                centroid_A_2D = self.cluster_info_A[self.cluster_info_A['Cluster'] == cluster_id]["2D_Centroid"].values[0]
                centroid_B_2D = row["2D_Centroid"]
                start_x, start_y = get_x(centroid_A_2D), get_y(centroid_A_2D)
                end_x, end_y = get_x(centroid_B_2D), get_y(centroid_B_2D)
                ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y), arrowprops=dict(arrowstyle="->", color="black", lw=2))
                drifts[cluster_id] = np.array(centroid_B_2D) - np.array(centroid_A_2D)
            else:
                drifts[cluster_id] = np.nan

        # --- Add text labels for clusters ---
        for idx, row in self.cluster_info_A.iterrows():
            cluster_id = row['Cluster']
            centroid_A_2D = self.cluster_info_A[self.cluster_info_A['Cluster'] == cluster_id]["2D_Centroid"].values[0]
            centroid_B_2D = self.cluster_info_B[self.cluster_info_B['Cluster'] == cluster_id]["2D_Centroid"].values[0]
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
            ScalarMappable(norm=norm_A, cmap=a_color), ax=ax, orientation="vertical", location="left", pad=0.15, ticks=all_labels
        ); cbar_A.set_label("Cluster ID (A)", fontsize=10)
        cbar_B = fig.colorbar(
            ScalarMappable(norm=norm_B, cmap=b_color), ax=ax, orientation="vertical", location="right", pad=0.15, ticks=all_labels
        ); cbar_B.set_label("Cluster ID (B)", fontsize=10)

        # --- Final styling ---
        legend_dot_A = mlines.Line2D([], [], color='#3787c0', marker='o', linestyle='None', markersize=6, label='Period A - Doc. Embeddings')
        legend_dot_B = mlines.Line2D([], [], color='#e32f27', marker='o', linestyle='None', markersize=6, label='Period B - Doc. Embeddings')
        ax.legend(handles=[legend_dot_A, legend_dot_B], loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
        return fig, ax


    def apply_modeling(self, df_A, df_B, n_top_freq_words=20, n_repr_docs=10, n_top_tf_idf_words=20):
        self.dataset_A = Dataset(df_A)
        self.dataset_B = Dataset(df_B)

        self._apply_umap()
        self._apply_leiden()
        self._exctract_info(n_top_freq_words, n_repr_docs, n_top_tf_idf_words)