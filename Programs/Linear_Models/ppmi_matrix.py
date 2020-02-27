import numpy as np
import math


def get_ppmi_matrix(ww_matrix, svd):  # get ppmi martix from co-occurrence matrix
    size = ww_matrix.shape
    ppmi_matrix = np.zeros(size)
    pmi_matrix = np.zeros(size)
    row_sum = ww_matrix.sum(1)
    column_sum = ww_matrix.sum(0)
    grand_sum = ww_matrix.sum()
    for i in range(size[0]):
        for j in range(size[1]):
            if ww_matrix[i][j] != 0:
                ppmi_matrix[i][j]=math.log2(ww_matrix[i][j]*grand_sum/(row_sum[i]*column_sum[j]))
                pmi_matrix[i][j] = ppmi_matrix[i][j]
                if ppmi_matrix[i][j] < 0:
                    ppmi_matrix[i][j] = 0
            else:
                ppmi_matrix[i][j] = 0
                pmi_matrix[i][j] = 0
    if svd:
        pmi_matrix = np.linalg.svd(ppmi_matrix)[0]
        ppmi_matrix = np.linalg.svd(ppmi_matrix)[0]
    return ppmi_matrix, pmi_matrix