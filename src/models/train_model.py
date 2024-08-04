from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_kmeans(df, n_clusters):
    kmodel = KMeans(n_clusters=n_clusters).fit(df[['Annual_Income', 'Spending_Score']])
    return kmodel

def calculate_wcss(df, k_range):
    WCSS = []
    K = []
    for i in k_range:
        kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income', 'Spending_Score']])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
    return K, WCSS

def calculate_silhouette(df, k_range):
    ss = []
    K = []
    for i in k_range:
        kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income', 'Spending_Score']])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Annual_Income', 'Spending_Score']], ypred)
        ss.append(sil_score)
        K.append(i)
    return K, ss