from sklearn.decomposition import PCA

def apply_pca(embeddings, n_components=256):
    # Apply PCA to reduce dimensionality of embeddings
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    return reduced_embeddings
