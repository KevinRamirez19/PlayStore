import os
import io
import base64
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ObtenerDatos():
    return [
        {"nombre": "Ana",    "edad": 22, "ingresos": 1200, "gasto": 300},
        {"nombre": "Luis",   "edad": 25, "ingresos": 1500, "gasto": 350},
        {"nombre": "Carlos", "edad": 23, "ingresos": 1300, "gasto": 280},
        {"nombre": "Marta",  "edad": 45, "ingresos": 4000, "gasto": 1200},
        {"nombre": "Sofía",  "edad": 50, "ingresos": 4200, "gasto": 1400},
        {"nombre": "Jorge",  "edad": 47, "ingresos": 3900, "gasto": 1100},
        {"nombre": "Elena",  "edad": 31, "ingresos": 2500, "gasto": 700},
        {"nombre": "Pedro",  "edad": 33, "ingresos": 2700, "gasto": 750},
        {"nombre": "Laura",  "edad": 29, "ingresos": 2400, "gasto": 680},
        {"nombre": "Andrés", "edad": 52, "ingresos": 5000, "gasto": 1600},
        {"nombre": "Camila", "edad": 21, "ingresos": 1100, "gasto": 250},
        {"nombre": "Diego",  "edad": 38, "ingresos": 3200, "gasto": 900},
    ]


def RealizarClustering(nclusters=3):
    datos = ObtenerDatos()
    X = [[p["edad"], p["ingresos"], p["gasto"]] for p in datos]

    scaler = StandardScaler()
    xScaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=nclusters, random_state=42, n_init=10)
    etiquetas = model.fit_predict(xScaled)

    resultados = []
    for i, persona in enumerate(datos):
        fila = persona.copy()
        fila["cluster"] = int(etiquetas[i])
        resultados.append(fila)

    resumenClusters = {}
    for etiqueta in etiquetas:
        etiqueta = int(etiqueta)
        resumenClusters[etiqueta] = resumenClusters.get(etiqueta, 0) + 1

    centroides = model.cluster_centers_.tolist()

    return {
        "resultados": resultados,
        "resumenClusters": resumenClusters,
        "centroides": centroides,
    }



BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "data", "playstore_limpio.csv")

CLUSTER_COLORS = ["#6C63FF", "#FF6584", "#43D9AD", "#FFB347", "#4FC3F7", "#F06292"]


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img


def _grafica_codo(X_scaled, k_optimo=4, max_k=10):
    ks       = list(range(1, max_k + 1))
    inercias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
                for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    ax.plot(ks, inercias, color="#6C63FF", linewidth=2.5, marker="o",
            markersize=8, markerfacecolor="#FF6584",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)

    ax.axvline(x=k_optimo, color="#43D9AD", linestyle="--",
               linewidth=1.5, alpha=0.85,
               label=f"K seleccionado (k={k_optimo})")
    ax.scatter([k_optimo], [inercias[k_optimo - 1]],
               color="#43D9AD", s=120, zorder=5,
               edgecolors="white", linewidth=1.5)

    ax.set_xlabel("Número de clústeres (K)", color="#CCCCDD", fontsize=11)
    ax.set_ylabel("Inercia (WCSS)",           color="#CCCCDD", fontsize=11)
    ax.set_title("Método del Codo — Selección de K óptimo",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="#CCCCDD")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.grid(True, color="#222244", linewidth=0.7, linestyle="--")
    ax.legend(facecolor="#1A1A2E", edgecolor="#333355",
              labelcolor="#CCCCDD", fontsize=10)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _grafica_scatter(df, centroides_orig, features, nclusters):
    from sklearn.decomposition import PCA

    X_num = df[features].values
    scaler_tmp = StandardScaler()
    X_sc = scaler_tmp.fit_transform(X_num)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)
    var1, var2 = pca.explained_variance_ratio_ * 100

    # Centroides en espacio PCA
    cent_num = np.array([[c[f] for f in features] for c in
                          [dict(zip(features, centroides_orig[i]))
                           for i in range(nclusters)]])
    cent_sc  = scaler_tmp.transform(cent_num)
    cent_pca = pca.transform(cent_sc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0F0F1A")

    ax1.set_facecolor("#0F0F1A")
    for c in range(nclusters):
        mask = df["Cluster"].values == c
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    alpha=0.6, s=28, label=f"Clúster {c}", zorder=2)
    for c in range(nclusters):
        ax1.scatter(cent_pca[c, 0], cent_pca[c, 1],
                    color="white", s=200, marker="*", zorder=5,
                    edgecolors=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    linewidth=1.5)
    ax1.set_xlabel(f"PC1 ({var1:.1f}% var.)", color="#CCCCDD", fontsize=10)
    ax1.set_ylabel(f"PC2 ({var2:.1f}% var.)", color="#CCCCDD", fontsize=10)
    ax1.set_title("PCA 2D — Separación Real de Clústeres",
                  color="white", fontsize=11, fontweight="bold", pad=12)
    ax1.tick_params(colors="#CCCCDD")
    for sp in ax1.spines.values(): sp.set_edgecolor("#333355")
    ax1.grid(True, color="#222244", linewidth=0.6, linestyle="--")
    ax1.legend(facecolor="#1A1A2E", edgecolor="#333355",
               labelcolor="#CCCCDD", fontsize=9)

    ax2.set_facecolor("#0F0F1A")
    for c in range(nclusters):
        sub = df[df["Cluster"] == c]
        ax2.scatter(sub["Price"],
                    sub["Rating"] + np.random.uniform(-0.04, 0.04, len(sub)),
                    color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    alpha=0.55, s=28, label=f"Clúster {c}", zorder=2)
    ip = features.index("Price")
    ir = features.index("Rating")
    for c in range(nclusters):
        ax2.scatter(centroides_orig[c][ip], centroides_orig[c][ir],
                    color="white", s=200, marker="*", zorder=5,
                    edgecolors=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    linewidth=1.5)
    ax2.set_xlabel("Price (USD)",  color="#CCCCDD", fontsize=10)
    ax2.set_ylabel("Rating",       color="#CCCCDD", fontsize=10)
    ax2.set_title("Price vs Rating — Distribución por Clúster",
                  color="white", fontsize=11, fontweight="bold", pad=12)
    ax2.tick_params(colors="#CCCCDD")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333355")
    ax2.grid(True, color="#222244", linewidth=0.6, linestyle="--")

    plt.tight_layout()
    return _fig_to_b64(fig)


def _grafica_centroides(centroides_df, features, nclusters):
    fig, axes = plt.subplots(1, len(features), figsize=(14, 5))
    fig.patch.set_facecolor("#0F0F1A")
    fig.suptitle("Valores de Centroides por Clúster y Variable",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    for i, feat in enumerate(features):
        ax   = axes[i]
        vals = centroides_df[feat].tolist()
        ax.set_facecolor("#0F0F1A")
        bars = ax.bar(range(nclusters), vals,
                      color=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                             for c in range(nclusters)],
                      edgecolor="#0F0F1A", linewidth=0.8)
        ax.set_title(feat, color="#CCCCDD", fontsize=10, fontweight="bold")
        ax.set_xticks(range(nclusters))
        ax.set_xticklabels([f"C{c}" for c in range(nclusters)],
                           color="#CCCCDD", fontsize=9)
        ax.tick_params(colors="#CCCCDD", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.grid(True, color="#222244", linewidth=0.7, axis="y", linestyle="--")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.1f}", ha="center", va="bottom",
                    color="white", fontsize=7)
    plt.tight_layout()
    return _fig_to_b64(fig)


def RealizarClusteringPlayStore(nclusters=4):
    df       = pd.read_csv(CSV_PATH)
    features = ["Rating", "Reviews", "Installs", "Price", "SizeMB"]
    X        = df[features].copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model    = KMeans(n_clusters=nclusters, random_state=42, n_init=10)
    etiquetas = model.fit_predict(X_scaled)
    df        = df.copy()
    df["Cluster"] = etiquetas

    # Resumen
    resumen = (df.groupby("Cluster")
                 .agg(Cantidad=("App","count"),
                      Rating_prom=("Rating","mean"),
                      Reviews_prom=("Reviews","mean"),
                      Installs_prom=("Installs","mean"),
                      Price_prom=("Price","mean"),
                      SizeMB_prom=("SizeMB","mean"))
                 .round(2).reset_index()
                 .to_dict(orient="records"))

    # Centroides escala original
    cent_orig = scaler.inverse_transform(model.cluster_centers_)
    cent_df   = pd.DataFrame(cent_orig, columns=features)
    cent_df.insert(0, "Cluster", range(nclusters))
    cent_df   = cent_df.round(2)

    # Imágenes
    img_codo       = _grafica_codo(X_scaled, k_optimo=nclusters)
    img_scatter    = _grafica_scatter(df, cent_orig, features, nclusters)
    img_centroides = _grafica_centroides(cent_df, features, nclusters)

    tabla = (df[["App","Category","Rating","Reviews",
                 "Installs","Price","SizeMB","Cluster"]]
               .head(200)
               .to_dict(orient="records"))

    return {
        "tabla":          tabla,
        "resumen":        resumen,
        "centroides":     cent_df.to_dict(orient="records"),
        "img_codo":       img_codo,
        "img_scatter":    img_scatter,
        "img_centroides": img_centroides,
        "inercia":        round(model.inertia_, 2),
        "nclusters":      nclusters,
        "total":          len(df),
    }