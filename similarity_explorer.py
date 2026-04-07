"""
===========================================================
Pattern Similarity Explorer
===========================================================

Author: Harinii Sathappan

Description:
This tool analyzes similarity between data samples using
cosine similarity and visualizes relationships as a heatmap.

It can be applied to scientific datasets such as biological,
environmental, or remote sensing data.

===========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_similarity_matrix(data):
    """
    Create similarity matrix for all samples
    """
    n = len(data)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(data[i], data[j])

    return matrix


def plot_heatmap(matrix):
    """
    Visualize similarity matrix
    """
    plt.figure(figsize=(6,6))
    plt.imshow(matrix, cmap='hot')
    plt.title("Similarity Heatmap")
    plt.colorbar()
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.show()


def main():
    print("=== Pattern Similarity Explorer ===")

    # Load dataset
    data = pd.read_csv("sample_data.csv")
    values = data.values

    # Compute similarity
    similarity_matrix = compute_similarity_matrix(values)

    # Display heatmap
    plot_heatmap(similarity_matrix)


if __name__ == "__main__":
    main()