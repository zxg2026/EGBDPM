import time
import numpy as np
from munkres import Munkres
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from GB_v2 import get_gb, get_DM_v2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix

def calculate_center_and_radius(gb):
    data_no_label = gb[:, :]
    center = data_no_label.mean(axis=0)
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    N = gb.shape[0]
    ball_quality = get_DM_v2(gb)
    return center, radius, N, ball_quality

def extract_ball_features_vectorized(gb_list):
    if not gb_list:
        return np.array([]), np.array([]), np.array([]), np.array([])
    features = [calculate_center_and_radius(gb) for gb in gb_list]
    centers, radiuss, ball_ms, ball_qualitys = zip(*features)
    return (np.array(centers), np.array(radiuss), np.array(ball_qualitys), np.array(ball_ms))

def ball_density(ball_qualitysA, ball_mA, centersA, k):
    base_density = np.where(ball_mA > 2, ball_qualitysA, 1000)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(centersA)
    _, k_nearest_indices = nbrs.kneighbors(centersA)
    neighbor_densities = base_density[k_nearest_indices]
    final_densities = (k+1) / np.sum(neighbor_densities, axis=1)
    return final_densities

def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0]
    ball_min_distAD = np.zeros(N3)
    ball_nearestAD = np.zeros(N3, dtype=int)
    index_ball_dens = np.argsort(-ball_densS)
    for i3 in range(1, N3):
        current_idx = index_ball_dens[i3]
        higher_indices = index_ball_dens[:i3]
        distances = ball_distS[current_idx, higher_indices]
        min_idx = np.argmin(distances)
        ball_min_distAD[current_idx] = distances[min_idx]
        ball_nearestAD[current_idx] = higher_indices[min_idx]
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD *= 10
    return ball_min_distAD, ball_nearestAD

def ball_draw_decision(ball_densS, ball_min_distS, k):
    rho = ball_densS * ball_min_distS
    return np.argsort(-rho)[:k]

def ball_cluster(ball_densS, ball_centers, ball_nearest):
    K1 = len(ball_centers)
    if K1 == 0:
        print('no centers')
        return
    N5 = ball_densS.shape[0]
    ball_labs = -np.ones(N5, dtype=int)
    ball_labs[ball_centers] = np.arange(1, K1 + 1)
    sorted_indices = np.argsort(-ball_densS)
    for idx in sorted_indices:
        if ball_labs[idx] == -1:
            ball_labs[idx] = ball_labs[ball_nearest[idx]]
    return ball_labs

def update_point_labels(data, ball_labs, gb_list):
    labels = -np.ones(data.shape[0], dtype=int)
    gb_dict = {}
    for i6 in range(len(gb_list)):
        for j6, point in enumerate(gb_list[i6]):
            gb_dict[tuple(point)] = ball_labs[i6]
    for i, data1 in enumerate(data):
        if tuple(data1) in gb_dict and labels[i] == -1:
            labels[i] = gb_dict[tuple(data1)]
    return labels

def evaluation(y_true, y_pred):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')
    return acc, nmi, ari, f1

def fit_sphere(points):
    A = np.hstack((2 * points, np.ones((points.shape[0], 1))))
    f = (points ** 2).sum(axis=1)
    C, resid, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:-1]
    radius = np.sqrt((center ** 2).sum() + C[-1])
    return center, radius

def sphere_arc_length(p1, p2, center, radius):
    v1 = p1 - center
    v2 = p2 - center
    theta = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return radius * theta

def build_spherelet_graph(X, k):
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    neighbors = nbrs.fit(X).kneighbors(X, return_distance=False)[:, 1:]
    G = lil_matrix((N, N))
    for i in range(N):
        local_idx = np.concatenate(([i], neighbors[i]))
        local_points = X[local_idx]
        center, radius = fit_sphere(local_points)
        for j in neighbors[i]:
            arc = sphere_arc_length(X[i], X[j], center, radius)
            G[i, j] = arc
            G[j, i] = arc
    return G.tocsr()

def estimate_spherelet_geodesic(X, k):
    G = build_spherelet_graph(X, k)
    geodesic_distances = dijkstra(G, directed=False, return_predecessors=False)
    return geodesic_distances

if __name__ == "__main__":

    file_names = ['Iris','pendigits']
    knn1 = [2, 9]
    Result_all = []
    for i, file_name in enumerate(file_names):
        df = np.loadtxt(f'{file_name}.txt')
        data = df[:, :-1]
        data_label = df[:, -1]
        k = len(set(data_label))
        knn = knn1[i]

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

        gb_list, gb_time = get_gb(data, 0, 0)  

        start = time.time()
        centersA, radiusA, ball_qualitysA, ball_mA = extract_ball_features_vectorized(gb_list)
        ball_densS = ball_density(ball_qualitysA, ball_mA, centersA, knn)
        ball_distS = estimate_spherelet_geodesic(centersA, knn)
        ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)
        ball_centers = ball_draw_decision(ball_densS, ball_min_distS, k)
        ball_labs = ball_cluster(ball_densS, ball_centers, ball_nearest)
        end = time.time()

        label = update_point_labels(data, ball_labs, gb_list)
        times = end - start + gb_time

        ACC, NMI, ARI, F1 = evaluation(data_label, label)
        print(f'数据集：{file_name}已完成')
        result = {
            'Dataset': file_name,
            'n': data.shape[0],
            'Knn': knn,
            'ACC': f'{ACC:.4f}',
            'NMI': f'{NMI:.4f}',
            'ARI': f'{ARI:.4f}',
            'Time': f'{times:.4f}'
            }
        Result_all.append(result)
    print(' ******************** 所有结果 ************************ ')
    if Result_all:
        df = pd.DataFrame(Result_all)
        df = df.astype(str)
        print(df.to_string(index=False))