import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import pdist, squareform

def division(hb_list):
    gb_list_new = []
    small_balls = []
    large_balls = []
    for hb in hb_list:
        if len(hb) <= 8:
            small_balls.append(hb)
        else:
            large_balls.append(hb)
    gb_list_new.extend(small_balls)
    for hb in large_balls:
        ball_1, ball_2 = spilt_ball_pca(hb)
        if len(ball_1) == 0 or len(ball_2) == 0:
            gb_list_new.append(hb)
            continue
        balls = [hb, ball_1, ball_2]
        DMs = [get_DM_v2(ball) for ball in balls]
        DM_parent, DM_child_1, DM_child_2 = DMs
        hb_len = len(hb)
        w_child = (len(ball_1) * DM_child_1 + len(ball_2) * DM_child_2) / hb_len
        if w_child < DM_parent:
            gb_list_new.extend([ball_1, ball_2])
        else:
            gb_list_new.append(hb)
    return gb_list_new

def get_DM_v2(hb):
    num = len(hb)
    if num <= 2:
        return np.inf
    center = np.mean(hb, axis=0)
    distances = np.linalg.norm(hb - center, axis=1)
    compactness = np.mean(distances)
    uniformity = np.std(distances) / (compactness + 1e-10)
    quality_measure = compactness * (1 + uniformity)
    return quality_measure

def spilt_ball(data):
    dist_matrix = squareform(pdist(data, metric='euclidean'))
    r, c = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    ball1 = []
    ball2 = []
    for i in range(len(data)):
        if dist_matrix[i, r] < dist_matrix[i, c]:
            ball1.append(data[i])
        else:
            ball2.append(data[i])
    return np.array(ball1), np.array(ball2)

def spilt_ball_pca(data):
    n, d = data.shape
    if n <= 2:
        return data[:1], data[1:] if n > 1 else np.array([]).reshape(0, d)
    center = np.mean(data, axis=0)
    centered_data = data - center
    if d >= 100:
        target_dim = min(50, d // 4, n // 2)
        np.random.seed(42)
        random_matrix = np.random.randn(d, target_dim) / np.sqrt(target_dim)
        projected_data = centered_data @ random_matrix
        U, _, _ = np.linalg.svd(projected_data.T, full_matrices=False)
        principal_component = random_matrix @ U[:, 0]
        principal_component /= np.linalg.norm(principal_component)
    else: 
        U, _, _ = np.linalg.svd(centered_data.T, full_matrices=False)
        principal_component = U[:, 0]
    projections = centered_data @ principal_component
    median_proj = np.median(projections)
    ball1_indices = projections <= median_proj
    ball1 = data[ball1_indices]
    ball2 = data[~ball1_indices]
    return ball1, ball2

def splits_1(data, num):
    ball_list = []
    if data.shape[0] < 7000:
        kmeans = KMeans(n_clusters=num, random_state=42, n_init=1, init='k-means++', max_iter=1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=num, random_state=42, n_init=1, init='k-means++', max_iter=1, batch_size=1000)
    label = kmeans.fit_predict(data)
    ball_list = [data[label == i] for i in range(num)]
    return ball_list

def calculate_center_and_radius(gb):
    data_no_label = gb[:, :]
    center = data_no_label.mean(axis=0)
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius

def gb_plot(gb_list, plt_type):
    plt.figure(figsize=(8, 6))
    plt.axis()
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)
        if plt_type == 0: 
            plt.plot(gb[:, 0], gb[:, 1], '.', c='k', markersize=5)
        if plt_type <= 3:
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, c='r', linewidth=1)
        if plt_type == 3:
            plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color='r',markersize=5)
    plt.yticks(color='k', fontsize=7)
    plt.xticks(color='k', fontsize=7)
    plt.tight_layout()
    plt.show()

def calculate_radius(gb):
    if len(gb) <= 1:
        return 0.0
    center = np.mean(gb, axis=0)
    distances = np.linalg.norm(gb - center, axis=1)
    return np.max(distances)

def normalized_ball(gb_list, threshold):
    if len(gb_list) == 0:
        return gb_list
    small_balls = []
    large_balls_info = []
    for hb in gb_list:
        if len(hb) <= 2:
            small_balls.append(hb)
        else:
            radius = calculate_radius(hb)
            large_balls_info.append((hb, radius))
    result_list = small_balls.copy()
    for hb, radius in large_balls_info:
        if radius <= threshold:
            result_list.append(hb)
        else:
            ball_1, ball_2 = spilt_ball_pca(hb)

            if len(ball_1) > 0 and len(ball_2) > 0:
                result_list.extend([ball_1, ball_2])
            else:
                result_list.append(hb)
    return result_list

def compute_threshold(gb_list):
    all_radii = []
    for hb in gb_list:
        radius = calculate_radius(hb)
        all_radii.append(radius)
    all_radii = np.array(all_radii)
    Q1, Q3 = np.percentile(all_radii, [25, 75])
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    threshold = np.clip(threshold, np.median(all_radii), np.percentile(all_radii,95))
    return threshold

def get_gb(X, plt, plt_type):
    start= time.time()
    num = int(np.ceil(np.sqrt(X.shape[0])))
    gb_list = splits_1(X, num=num)
    start1 = time.time()
    if plt == 1:
        gb_plot(gb_list, plt_type)
    end1 = time.time()
    while 1:
        ball_number_1 = len(gb_list)
        gb_list = division(gb_list)
        ball_number_2 = len(gb_list)
        if ball_number_1 == ball_number_2:
            break
    start2 = time.time()
    if plt == 1 or plt == 2:
        gb_plot(gb_list, plt_type)
    end2 = time.time()
    threshold = compute_threshold(gb_list)
    while 1:
        ball_number_old = len(gb_list)
        gb_list = normalized_ball(gb_list, threshold)
        ball_number_new = len(gb_list)
        if ball_number_new == ball_number_old:
            break
    start3 = time.time()
    if plt == 3:
        gb_plot(gb_list, plt_type)
    end3 = time.time()
    end = time.time()
    times = end - start - (end1 - start1) - (end2 - start2) - (end3 - start3)
    return gb_list, times


if __name__ == "__main__":
    num = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for n in num:
        print("当前样本数：", n)
        X, _ = make_blobs(n_samples=n, centers=4, n_features=2, random_state=42)
        X = X.astype(np.float32)  
        times = []
        for i in range(1):
            start = time.time()
            gb_list = get_gb(X)
            end = time.time()
            if i == 0:
                times.append(end - start)
        avg_time = np.mean(times)
        print("优化后粒球生成平均时间：%.5f" % avg_time)
        #gb_plot(gb_list, plt_type=0)
        print("\n")