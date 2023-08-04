import numpy as np
from sklearn.decomposition import PCA

"""
calculates the longest diagonal of a polygon using PCA algorithm.
input should be an opencv contour for a single object.
"""
def calc_longest_diagonal_pca(contour):

    contour = np.squeeze(contour)
    
    pca = PCA(n_components=1)
    pca.fit(contour)

    principal_component = pca.components_[0]
    contour_pca = np.dot(contour, principal_component)

    start_index = np.argmin(contour_pca)
    end_index = np.argmax(contour_pca)

    start, end = contour[start_index], contour[end_index]
    start, end = tuple(start), tuple(end)
    length = math.dist(start, end)

    return start, end, length