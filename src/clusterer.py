"""
Clustering Module
Handles clustering of similar QA pairs using DBSCAN
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from typing import List, Dict, Any, Tuple
import streamlit as st

def cluster_qa_pairs(
    qa_pairs_with_embeddings: List[Dict[str, Any]],
    eps: float = 0.3,
    min_samples: int = 2,
    metric: str = 'cosine'
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Cluster QA pairs using DBSCAN
    
    Args:
        qa_pairs_with_embeddings: QA pairs with embeddings
        eps: Maximum distance between samples in a cluster
        min_samples: Minimum samples in a cluster
        metric: Distance metric (cosine, euclidean)
        
    Returns:
        Tuple of (clustered QA pairs, clustering stats)
    """
    try:
        if not qa_pairs_with_embeddings:
            return [], {}
        
        # Extract embeddings
        embeddings = np.array([qa['embedding'] for qa in qa_pairs_with_embeddings])
        
        # Perform DBSCAN clustering
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Add cluster labels to QA pairs
        clustered_qa_pairs = []
        for qa, label in zip(qa_pairs_with_embeddings, cluster_labels):
            qa_with_cluster = qa.copy()
            qa_with_cluster['cluster'] = int(label)
            clustered_qa_pairs.append(qa_with_cluster)
        
        # Calculate statistics
        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate silhouette score (only if we have clusters)
        silhouette = None
        if n_clusters > 1 and n_noise < len(cluster_labels):
            try:
                # Remove noise points for silhouette calculation
                valid_indices = cluster_labels != -1
                if sum(valid_indices) > 1:
                    silhouette = silhouette_score(
                        embeddings[valid_indices],
                        cluster_labels[valid_indices],
                        metric=metric
                    )
            except:
                silhouette = None
        
        # Group by cluster
        clusters_dict = {}
        for qa in clustered_qa_pairs:
            cluster_id = qa['cluster']
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(qa)
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_total': len(qa_pairs_with_embeddings),
            'n_clustered': len(qa_pairs_with_embeddings) - n_noise,
            'silhouette_score': silhouette,
            'clusters_dict': clusters_dict,
            'cluster_sizes': {k: len(v) for k, v in clusters_dict.items() if k != -1}
        }
        
        return clustered_qa_pairs, stats
    
    except Exception as e:
        raise Exception(f"Error clustering QA pairs: {str(e)}")


def find_optimal_eps(
    embeddings: np.ndarray,
    min_samples: int = 2,
    metric: str = 'cosine'
) -> float:
    """
    Find optimal epsilon value for DBSCAN
    Uses k-distance graph method
    
    Args:
        embeddings: Array of embeddings
        min_samples: Minimum samples parameter
        metric: Distance metric
        
    Returns:
        Suggested epsilon value
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        # Fit nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric=metric)
        neighbors.fit(embeddings)
        distances, indices = neighbors.kneighbors(embeddings)
        
        # Sort distances
        distances = np.sort(distances[:, -1])
        
        # Find elbow point (simplified - take 90th percentile)
        suggested_eps = np.percentile(distances, 90)
        
        return float(suggested_eps)
    
    except Exception as e:
        # Return default if calculation fails
        return 0.3


def get_cluster_representatives(
    clusters_dict: Dict[int, List[Dict[str, Any]]],
    method: str = 'centroid'
) -> List[Dict[str, Any]]:
    """
    Select representative QA pairs from each cluster
    
    Args:
        clusters_dict: Dictionary mapping cluster IDs to QA pairs
        method: Selection method ('centroid', 'first', 'longest')
        
    Returns:
        List of representative QA pairs
    """
    representatives = []
    
    for cluster_id, qa_pairs in clusters_dict.items():
        if cluster_id == -1:  # Skip noise
            continue
        
        if not qa_pairs:
            continue
        
        if method == 'centroid':
            # Find QA pair closest to cluster centroid
            embeddings = np.array([qa['embedding'] for qa in qa_pairs])
            centroid = np.mean(embeddings, axis=0)
            
            # Calculate distances to centroid
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            representative_idx = np.argmin(distances)
            representative = qa_pairs[representative_idx].copy()
        
        elif method == 'first':
            # Take first QA pair
            representative = qa_pairs[0].copy()
        
        elif method == 'longest':
            # Take QA pair with longest answer
            representative = max(qa_pairs, key=lambda x: len(x.get('answer', ''))).copy()
        
        else:
            representative = qa_pairs[0].copy()
        
        # Add metadata
        representative['cluster_id'] = cluster_id
        representative['cluster_size'] = len(qa_pairs)
        representative['is_representative'] = True
        
        representatives.append(representative)
    
    # Add noise points as individual representatives
    if -1 in clusters_dict:
        for qa in clusters_dict[-1]:
            qa_copy = qa.copy()
            qa_copy['cluster_id'] = -1
            qa_copy['cluster_size'] = 1
            qa_copy['is_representative'] = True
            representatives.append(qa_copy)
    
    return representatives


def visualize_clusters_2d(
    qa_pairs_with_embeddings: List[Dict[str, Any]],
    method: str = 'umap'
) -> Tuple[np.ndarray, List[int]]:
    """
    Reduce embeddings to 2D for visualization
    
    Args:
        qa_pairs_with_embeddings: QA pairs with embeddings and cluster labels
        method: Dimensionality reduction method ('umap' or 'tsne')
        
    Returns:
        Tuple of (2D coordinates, cluster labels)
    """
    try:
        embeddings = np.array([qa['embedding'] for qa in qa_pairs_with_embeddings])
        cluster_labels = [qa.get('cluster', -1) for qa in qa_pairs_with_embeddings]
        
        if method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                coords_2d = reducer.fit_transform(embeddings)
            except ImportError:
                st.warning("UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(embeddings)
        
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            coords_2d = reducer.fit_transform(embeddings)
        
        else:  # PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)
        
        return coords_2d, cluster_labels
    
    except Exception as e:
        st.error(f"Error in dimensionality reduction: {str(e)}")
        return np.array([]), []


def analyze_cluster_quality(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze clustering quality and provide insights
    
    Args:
        stats: Clustering statistics
        
    Returns:
        Quality analysis dictionary
    """
    analysis = {
        'quality': 'Unknown',
        'insights': [],
        'recommendations': []
    }
    
    # Silhouette score interpretation
    silhouette = stats.get('silhouette_score')
    if silhouette is not None:
        if silhouette > 0.7:
            analysis['quality'] = 'Excellent'
            analysis['insights'].append(f"High silhouette score ({silhouette:.2f}) indicates well-separated clusters")
        elif silhouette > 0.5:
            analysis['quality'] = 'Good'
            analysis['insights'].append(f"Good silhouette score ({silhouette:.2f}) indicates reasonable clustering")
        elif silhouette > 0.3:
            analysis['quality'] = 'Fair'
            analysis['insights'].append(f"Fair silhouette score ({silhouette:.2f}) indicates some overlap between clusters")
            analysis['recommendations'].append("Consider adjusting eps parameter")
        else:
            analysis['quality'] = 'Poor'
            analysis['insights'].append(f"Low silhouette score ({silhouette:.2f}) indicates weak clustering")
            analysis['recommendations'].append("Try different eps or min_samples values")
    
    # Noise analysis
    n_noise = stats.get('n_noise', 0)
    n_total = stats.get('n_total', 1)
    noise_ratio = n_noise / n_total if n_total > 0 else 0
    
    if noise_ratio > 0.5:
        analysis['insights'].append(f"High noise ratio ({noise_ratio:.1%}) - many QA pairs don't fit in clusters")
        analysis['recommendations'].append("Consider decreasing eps to form tighter clusters")
    elif noise_ratio > 0.3:
        analysis['insights'].append(f"Moderate noise ratio ({noise_ratio:.1%})")
    else:
        analysis['insights'].append(f"Low noise ratio ({noise_ratio:.1%}) - most QA pairs are clustered")
    
    # Cluster size distribution
    cluster_sizes = stats.get('cluster_sizes', {})
    if cluster_sizes:
        avg_size = np.mean(list(cluster_sizes.values()))
        max_size = max(cluster_sizes.values())
        
        analysis['insights'].append(f"Average cluster size: {avg_size:.1f} QA pairs")
        
        if max_size > avg_size * 3:
            analysis['insights'].append(f"Some very large clusters detected (max: {max_size})")
            analysis['recommendations'].append("Large clusters might need sub-clustering")
    
    return analysis
