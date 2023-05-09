import numpy as np
from sklearn.cluster import KMeans


class Spectral_clustering:
    def __init__(self, n_clusters, sigma=1):
        self.sigma = sigma
        self.n_clusters = n_clusters

    def gaussian_affinity_matrix(self, X):
        distances = np.linalg.norm(X[:, None] - X, axis=2)
        return np.exp(-distances ** 2 / 2 * self.sigma ** 2)

    def Laplacian(self, A):
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        L = np.identity(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    def eigenvector(self, L):
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        return eigenvectors[:, :self.n_clusters]

    def fit_predict(self, X):
        affinity_matrix = self.gaussian_affinity_matrix(X)
        laplacian_matrix = self.Laplacian(affinity_matrix)
        eigenvectors = self.eigenvector(laplacian_matrix)
        model = KMeans(n_clusters=self.n_clusters)
        labels = model.fit_predict(eigenvectors)
        return labels


#from sklearn.datasets import make_blobs

#X, _ = make_blobs(n_samples=200, centers=5, random_state=42)

#model = Spectral_clustering(n_clusters=5, sigma=1.0)
#labels = model.fit_predict(X)