import numpy as np
import pandas as pd

def load_adjacency(dist_csv, ch_names, threshold_pct=75):
    """Read the 3-columns [from,to,distance] of distances_3d.csv and build a symmetric adjacency:"""
    # read and pivot
    df = pd.read_csv(dist_csv)
    dmat = df.pivot(index="from", columns="to", values="distance")
    dmat = dmat.reindex(index=ch_names, columns=ch_names)
    dist = dmat.values.astype(float)

    # zero the diagonal
    np.fill_diagonal(dist, 0.0)

    # mirror known entries to get symmetric matrix
    mask = np.isnan(dist)
    dist[mask] = dist.T[mask]

    # fill any remaining NaNs with the max so that missing pairs become “very far apart”
    max_dist = np.nanmax(dist)
    dist[np.isnan(dist)] = max_dist

    # build RBF weights
    sigma = dist.mean()
    W = np.exp(-(dist**2) / (2 * sigma**2))

    # sparsify by zeroing out the weakest edges
    cutoff = np.percentile(W, threshold_pct)
    W[W < cutoff] = 0.0

    # zero the diagonal again
    np.fill_diagonal(W, 0.0)

    return W