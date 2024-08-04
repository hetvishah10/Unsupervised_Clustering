import matplotlib.pyplot as plt
import seaborn as sns

def plot_elbow(K, WCSS):
    plt.plot(K, WCSS)
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()

def plot_silhouette(K, ss):
    plt.plot(K, ss)
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    plt.show()

def plot_clusters(df):
    sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.show()