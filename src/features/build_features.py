import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(df):
    sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']])
    plt.show()

def add_cluster_labels(df, kmodel):
    df['Cluster'] = kmodel.labels_
    return df