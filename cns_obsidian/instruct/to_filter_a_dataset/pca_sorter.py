import numpy as np


class PCASorter:
    def __init__(self):
        self.mean_image = None
        self.eigenvectors = None
        self.sorted_indices = None

    def fit(self, transformed_images):
        # Step 1: Compute the mean of the dataset
        self.mean_image = np.mean(transformed_images, axis=0)

        # Step 2: Center the dataset by subtracting the mean
        centered_images = transformed_images - self.mean_image

        # Step 3: Compute the covariance matrix of the centered dataset
        cov_matrix = np.cov(centered_images, rowvar=False)

        # Step 4: Compute the eigenvalues and eigenvectors of the covariance
        eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 5: Project the dataset onto the eigenvectors
        pca_components = np.dot(centered_images, self.eigenvectors)

        # Step 6: Sort the images by the first principal component
        self.sorted_indices = np.argsort(
            pca_components[:, -1]
        )  # Use last component since np.linalg.eigh sorts evals in asc order

    def transform(self, array):
        if self.sorted_indices is None:
            raise ValueError(
                "You must fit the PCA sorter before calling transform."
            )
        return [array[i] for i in self.sorted_indices]
