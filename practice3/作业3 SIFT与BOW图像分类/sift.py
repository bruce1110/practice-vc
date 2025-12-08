import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# -------------------- 配置区（按需修改） --------------------
dataset_root = "15-Scene"                # Scene15 数据集根目录
codebook_sizes = [10,15,64, 128, 256, 512, 1024]# 要测试的 codebook 大小
knn_k_list = [1, 3, 5, 8, 12,15]             # K-NN 中要测试的 K 值
svm_C_list = [0.01, 0.1, 1, 10, 100]     # SVM C 候选
svm_gamma_list = ['scale', 'auto']       # RBF gamma 候选
dense_step = 8                           # Dense SIFT 步长
max_desc_for_kmeans = 100000             # KMeans 描述子最大采样数
random_state = 42
cv_folds = 5
save_results_file = "bovw_knn_svm_results.joblib"
save_plot_prefix = "bovw_plot"           # 曲线图保存前缀
# ---------------------------------------------------------

# -------------------- 工具函数 ----------------------------
def list_images_and_labels(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    imgs, labels = [], []
    for idx, cls in enumerate(classes):
        dirc = os.path.join(root, cls)
        for f in os.listdir(dirc):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                imgs.append(os.path.join(dirc, f))
                labels.append(idx)
    return imgs, np.array(labels), classes

def extract_dense_sift(img_gray, sift, step=dense_step):
    """
    Dense SIFT 特征提取（网格采样）
    img_gray: 灰度图像
    sift: cv2.SIFT_create() 对象
    step: 网格步长
    返回 Nx128 描述子矩阵，若没有关键点返回 None
    """
    h, w = img_gray.shape[:2]
    # 使用 OpenCV 4.x 的 KeyPoint 构造函数（第三个参数直接为 size）
    kps = [cv2.KeyPoint(float(x), float(y), step) for y in range(0, h, step) for x in range(0, w, step)]
    if len(kps) == 0:
        return None
    kps, des = sift.compute(img_gray, kps)
    return des

def extract_sparse_sift(img_gray, sift):
    kps, des = sift.detectAndCompute(img_gray, None)
    return des

def extract_descriptors_for_images(image_paths, mode='dense', step=dense_step, verbose=True):
    sift = cv2.SIFT_create()
    descs = []
    iterator = tqdm(image_paths) if verbose else image_paths
    for p in iterator:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            descs.append(None)
            continue
        if mode == 'dense':
            des = extract_dense_sift(img, sift, step=step)
        else:
            des = extract_sparse_sift(img, sift)
        descs.append(des if (des is not None and len(des) > 0) else None)
    return descs

def sample_descriptors(descs_list, max_samples=max_desc_for_kmeans, rng_seed=random_state):
    # 将所有有效描述子堆叠并随机子采样
    all_desc = np.vstack([d for d in descs_list if d is not None and len(d) > 0])
    if all_desc.shape[0] > max_samples:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(all_desc.shape[0], size=max_samples, replace=False)
        all_desc = all_desc[idx]
    return all_desc.astype(np.float32)

def train_kmeans(train_descs_per_img, k):
    samples = sample_descriptors([d for d in train_descs_per_img if d is not None])
    print(f"    KMeans训练样本数: {samples.shape[0]}")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2000, random_state=random_state)
    kmeans.fit(samples)
    return kmeans

def compute_bow_hist(descriptors, kmeans, k):
    if descriptors is None or len(descriptors) == 0:
        hist = np.zeros(k, dtype=np.float32)
    else:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(k+1))
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
    return hist

# -------------------- 实验主流程 ----------------------------
def experiment_mode(image_paths, labels, mode):
    """
    mode: 'dense' or 'sparse'
    返回一个 results 字典，包含 knn 与 svm 的测试准确率（按 codebook_sizes 顺序）
    """
    print(f"\n===== 开始实验 mode = {mode} =====")
    idx = np.arange(len(image_paths))
    train_idx, test_idx, y_train, y_test = train_test_split(idx, labels, test_size=0.5,
                                                            stratify=labels, random_state=random_state)
    # 提取描述子（所有图片）
    print("提取 SIFT 描述子（这一步可能耗时）...")
    descs_per_img = extract_descriptors_for_images(image_paths, mode=mode, step=dense_step, verbose=True)

    results = {
        'codebook_sizes': [],
        'knn': {k_nn: [] for k_nn in knn_k_list},       # 每个 k_nn 对应一个 list，按 codebook_sizes 顺序
        'svm': {'linear': [], 'rbf': []},               # 每种 kernel 对应一个 list
        'svm_best_params': {},                          # 记录每个 k 的最佳参数和 cv score
        'knn_cv_info': {}                               # 记录每个 k 的训练集 CV 信息
    }

    for k in codebook_sizes:
        print(f"\n--- codebook k = {k} ---")
        results['codebook_sizes'].append(k)

        # 用训练集描述子训练 KMeans
        train_descs = [descs_per_img[i] for i in train_idx if descs_per_img[i] is not None]
        if len(train_descs) == 0:
            raise RuntimeError("训练集中没有有效描述子，检查数据集或提取参数。")
        kmeans = train_kmeans(train_descs, k)

        # 构建所有图像的 BoW
        print("  构建 BoW 直方图...")
        H = np.vstack([compute_bow_hist(descs_per_img[i], kmeans, k) for i in range(len(image_paths))])

        # 标准化（fit on train only）
        scaler = StandardScaler().fit(H[train_idx])
        Hs = scaler.transform(H)

        X_tr, X_te = Hs[train_idx], Hs[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        # ---------- K-NN: 对每个 K 值在训练集上 CV（参考）并测试 ----------
        knn_info_per_k = {}
        for k_nn in knn_k_list:
            print(f"  K-NN (k={k_nn}) - 在训练集上做 {cv_folds}-折 CV 参考...")
            knn = KNeighborsClassifier(n_neighbors=k_nn, n_jobs=-1)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(knn, X_tr, y_tr, cv=cv, n_jobs=-1, scoring='accuracy')
            cv_mean = float(np.mean(cv_scores))
            # 训练并测试
            knn.fit(X_tr, y_tr)
            ypred = knn.predict(X_te)
            test_acc = float(accuracy_score(y_te, ypred))
            print(f"    train-CV mean={cv_mean:.4f}, test_acc={test_acc:.4f}")
            results['knn'][k_nn].append(test_acc)
            knn_info_per_k[k_nn] = {'cv_mean': cv_mean, 'test_acc': test_acc}
        results['knn_cv_info'][k] = knn_info_per_k

        # ---------- SVM: linear kernel ----------
        print("  SVM (linear) - GridSearchCV 在训练集上调参...")
        param_grid_lin = {'C': svm_C_list}
        svc_lin = SVC(kernel='linear')
        grid_lin = GridSearchCV(svc_lin, param_grid_lin,
                                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                                scoring='accuracy', n_jobs=-1)
        grid_lin.fit(X_tr, y_tr)
        best_lin = grid_lin.best_estimator_
        lin_cv_best = float(grid_lin.best_score_)
        ypred_lin = best_lin.predict(X_te)
        test_lin = float(accuracy_score(y_te, ypred_lin))
        print(f"    linear best C={grid_lin.best_params_['C']}, train-CV={lin_cv_best:.4f}, test={test_lin:.4f}")
        results['svm']['linear'].append(test_lin)

        # ---------- SVM: rbf kernel ----------
        print("  SVM (rbf) - GridSearchCV 在训练集上调参...")
        param_grid_rbf = {'C': svm_C_list, 'gamma': svm_gamma_list}
        svc_rbf = SVC(kernel='rbf')
        grid_rbf = GridSearchCV(svc_rbf, param_grid_rbf,
                                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                                scoring='accuracy', n_jobs=-1)
        grid_rbf.fit(X_tr, y_tr)
        best_rbf = grid_rbf.best_estimator_
        rbf_cv_best = float(grid_rbf.best_score_)
        ypred_rbf = best_rbf.predict(X_te)
        test_rbf = float(accuracy_score(y_te, ypred_rbf))
        print(f"    rbf best params={grid_rbf.best_params_}, train-CV={rbf_cv_best:.4f}, test={test_rbf:.4f}")
        results['svm']['rbf'].append(test_rbf)

        # 记录最佳参数信息
        results['svm_best_params'][k] = {
            'linear': {'best_params': grid_lin.best_params_, 'cv_best': lin_cv_best, 'test_acc': test_lin},
            'rbf': {'best_params': grid_rbf.best_params_, 'cv_best': rbf_cv_best, 'test_acc': test_rbf}
        }

    return results

# -------------------- 主入口 ----------------------------
def main():
    print("加载数据集...")
    image_paths, labels, classes = list_images_and_labels(dataset_root)
    print(f"发现 {len(image_paths)} 张图片，{len(classes)} 个类别。")

    # Dense 实验
    res_dense = experiment_mode(image_paths, labels, mode='dense')

    # Sparse 实验
    res_sparse = experiment_mode(image_paths, labels, mode='sparse')

    # 保存结果
    all_results = {
        'dense': res_dense,
        'sparse': res_sparse,
        'codebook_sizes': codebook_sizes,
        'knn_k_list': knn_k_list,
        'svm_C_list': svm_C_list,
        'svm_gamma_list': svm_gamma_list,
        'classes': classes
    }
    joblib.dump(all_results, save_results_file)
    print(f"\n实验结果已保存到 {save_results_file}")

    # -------------------- 绘图：K-NN 多个 K 的对比（Dense / Sparse） --------------------
    plt.figure(figsize=(10, 6))
    for k_nn in knn_k_list:
        dense_vals = res_dense['knn'][k_nn]
        sparse_vals = res_sparse['knn'][k_nn]
        plt.plot(codebook_sizes, dense_vals, marker='o', linestyle='-', label=f'Dense KNN k={k_nn}')
        plt.plot(codebook_sizes, sparse_vals, marker='s', linestyle='--', label=f'Sparse KNN k={k_nn}')
    plt.xlabel('Codebook size (k)')
    plt.ylabel('Test accuracy')
    plt.title('K-NN: Codebook size vs Test Accuracy (different k)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    knn_plot_file = f"{save_plot_prefix}_knn.png"
    plt.savefig(knn_plot_file, dpi=150)
    print(f"K-NN 曲线图已保存到 {knn_plot_file}")
    plt.show()

    # -------------------- 绘图：SVM linear vs rbf 对比（Dense / Sparse） --------------------
    plt.figure(figsize=(8, 6))
    plt.plot(codebook_sizes, res_dense['svm']['linear'], marker='o', linestyle='-', label='Dense SVM (linear)')
    plt.plot(codebook_sizes, res_dense['svm']['rbf'], marker='o', linestyle='--', label='Dense SVM (rbf)')
    plt.plot(codebook_sizes, res_sparse['svm']['linear'], marker='s', linestyle='-', label='Sparse SVM (linear)')
    plt.plot(codebook_sizes, res_sparse['svm']['rbf'], marker='s', linestyle='--', label='Sparse SVM (rbf)')
    plt.xlabel('Codebook size (k)')
    plt.ylabel('Test accuracy')
    plt.title('SVM: Codebook size vs Test Accuracy (linear vs rbf)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    svm_plot_file = f"{save_plot_prefix}_svm.png"
    plt.savefig(svm_plot_file, dpi=150)
    print(f"SVM 曲线图已保存到 {svm_plot_file}")
    plt.show()

    # -------------------- 打印总结表格 --------------------
    print("\n===== 实验总结（按 codebook size 列出） =====")
    for i, k in enumerate(codebook_sizes):
        print(f"\n--- k = {k} ---")
        print("K-NN 测试集准确率：")
        for k_nn in knn_k_list:
            dval = res_dense['knn'][k_nn][i]
            sval = res_sparse['knn'][k_nn][i]
            print(f"  k={k_nn}: Dense={dval:.4f} | Sparse={sval:.4f}")
        print("SVM 测试集准确率：")
        print(f"  Dense linear={res_dense['svm']['linear'][i]:.4f}, Dense rbf={res_dense['svm']['rbf'][i]:.4f}")
        print(f"  Sparse linear={res_sparse['svm']['linear'][i]:.4f}, Sparse rbf={res_sparse['svm']['rbf'][i]:.4f}")

if __name__ == "__main__":
    main()
