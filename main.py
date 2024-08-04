from src.data.make_dataset import load_data, describe_data, show_data_head
from src.features.build_features import plot_pairplot, add_cluster_labels
from src.models.train_model import train_kmeans, calculate_wcss, calculate_silhouette
from src.visualization.visualize import plot_elbow, plot_silhouette, plot_clusters

def main():
    # Load the data
    df = load_data('data/raw/mall_customers.csv')

    # Explore the data
    describe_data(df)
    show_data_head(df)

    # Plot pairplot
    plot_pairplot(df)

    # Train KMeans model and analyze clusters
    kmodel = train_kmeans(df, n_clusters=5)
    df = add_cluster_labels(df, kmodel)

    # Visualize clusters
    plot_clusters(df)

    # Calculate and plot WCSS (Elbow method)
    K, WCSS = calculate_wcss(df, k_range=range(3, 9))
    plot_elbow(K, WCSS)

    # Calculate and plot Silhouette Scores
    K, ss = calculate_silhouette(df, k_range=range(3, 9))
    plot_silhouette(K, ss)

if __name__ == "__main__":
    main()