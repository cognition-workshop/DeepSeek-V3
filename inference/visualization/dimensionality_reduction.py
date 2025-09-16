import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Optional, Dict, Any

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimensionalityReducer:
    """
    Handles dimensionality reduction for high-dimensional embeddings.
    
    Supports t-SNE, UMAP, and PCA methods for reducing embeddings
    to 2D or 3D for visualization purposes.
    """
    
    def __init__(self):
        self.fitted_models: Dict[str, Any] = {}
        
    def reduce_tsne(self, embeddings: np.ndarray, n_components: int = 2, 
                   perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings array
            n_components: Number of output dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced embeddings array
        """
        if embeddings.shape[0] < perplexity * 3:
            perplexity = max(5, embeddings.shape[0] // 3)
            
        tsne = TSNE(
            n_components=n_components, 
            perplexity=perplexity, 
            random_state=random_state,
            max_iter=1000,
            learning_rate='auto',
            init='random'
        )
        reduced = tsne.fit_transform(embeddings)
        self.fitted_models['tsne'] = tsne
        return reduced
        
    def reduce_umap(self, embeddings: np.ndarray, n_components: int = 2, 
                   n_neighbors: int = 15, min_dist: float = 0.1, 
                   random_state: int = 42) -> np.ndarray:
        """
        Apply UMAP dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings array
            n_components: Number of output dimensions (2 or 3)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced embeddings array
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        n_neighbors = min(n_neighbors, embeddings.shape[0] - 1)
        
        reducer = umap.UMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='cosine'
        )
        reduced = reducer.fit_transform(embeddings)
        self.fitted_models['umap'] = reducer
        return reduced
        
    def reduce_pca(self, embeddings: np.ndarray, n_components: int = 2, 
                  random_state: int = 42) -> np.ndarray:
        """
        Apply PCA dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings array
            n_components: Number of output dimensions
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced embeddings array
        """
        n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
        
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced = pca.fit_transform(embeddings)
        self.fitted_models['pca'] = pca
        return reduced
    
    def get_explained_variance_ratio(self, method: str) -> Optional[np.ndarray]:
        """Get explained variance ratio for PCA method."""
        if method == 'pca' and 'pca' in self.fitted_models:
            return self.fitted_models['pca'].explained_variance_ratio_
        return None
    
    def get_available_methods(self) -> list:
        """Get list of available dimensionality reduction methods."""
        methods = ['tsne', 'pca']
        if UMAP_AVAILABLE:
            methods.append('umap')
        return methods
