import os
import matplotlib.pyplot as plt

def generate_graphs(df, graph_save_path):
    """
    Generate graphs for numeric and categorical columns in the DataFrame.
    :param df: DataFrame containing the data
    :param graph_save_path: Directory to save the generated graphs
    :return: List of generated graph file names
    """
    os.makedirs(graph_save_path, exist_ok=True)
    graph_files = []

    # Numeric Graphs
    numeric_columns = [
        col for col in df.select_dtypes(include='number').columns
        if col not in ['Cluster', 'Cluster Label']
    ]
    for col in numeric_columns:
        plt.figure(figsize=(8, 5))
        df.boxplot(column=col, by='Cluster', grid=False)
        plt.title(f"{col} Distribution by Cluster")
        plt.suptitle('')  # Suppress the default title
        plt.xlabel("Cluster ID")
        plt.ylabel(col)
        plt.tight_layout()

        sanitized_col_name = col.replace(' ', '_').lower()
        file_name = f"{sanitized_col_name}_distribution.png"
        plt.savefig(os.path.join(graph_save_path, file_name))
        plt.close()
        graph_files.append(file_name)

    # Categorical Graphs
    categorical_columns = [
        col for col in df.select_dtypes(include='object').columns
        if col not in ['Cluster', 'Cluster Label']
    ]
    for col in categorical_columns:
        plt.figure(figsize=(8, 5))
        df.groupby(['Cluster', col]).size().unstack().plot(kind='bar', stacked=True)
        plt.title(f"Composition of {col} by Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Count")
        plt.tight_layout()

        sanitized_col_name = col.replace(' ', '_').lower()
        file_name = f"{sanitized_col_name}_distribution.png"
        plt.savefig(os.path.join(graph_save_path, file_name))
        plt.close()
        graph_files.append(file_name)

    return graph_files
