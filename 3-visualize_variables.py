import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
file_path = "data/vannkraft_data.txt"
df = pd.read_csv(file_path, sep="\t")  # Assuming tab-separated values

# Display basic info
print(df.info())
print(df.head())

# Perform PCA on all x features
X = df.iloc[:, :-4].values
pca = PCA(n_components=X.shape[1])
pca.fit(X)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()


import pandas as pd

from sklearn.decomposition import PCA


# Extract input features (assuming df is already loaded)

X = df.iloc[:, :-4].values  # All x features


# Apply PCA with 5 components (since 94% variance is explained)

pca = PCA(n_components=5)

X_pca = pca.fit_transform(X)


# Get PCA component loadings (how original features contribute to PCs)

loadings = pd.DataFrame(
    pca.components_.T,
    index=df.columns[:-4],
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
)

print(loadings)
