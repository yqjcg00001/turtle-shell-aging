# ============================================================
# step2_二分类.py —— 训练3个模型：光谱 / 图像 / 多模态
# ============================================================
# 为什么用二分类？
#   当前 14 个样品分 4 类（1:9:3:1），类别极度不平衡
#   改成二分类后：早期（1+2级，10个） vs 晚期（3+4级，4个）
#   类别更均衡，模型更容易学到有效规律
#
# 运行: python step2_二分类.py
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config

# ============================================================
# 二分类映射
# ============================================================
BINARY_MAP = {
    1: "早期老化",
    2: "早期老化",
    3: "晚期老化",
    4: "晚期老化",
}

# ============================================================
# 模型配置列表
# ============================================================
MODEL_CONFIGS = [
    {
        "name": "光谱模型",
        "prefix": "spectral",
        "col_filter": lambda c: c.startswith("xrd_") or c.startswith("ir_"),
        "mi_k": 15,
        "rfe_cv": 3,
        "rfe_min": 2,
    },
    {
        "name": "图像模型",
        "prefix": "image",
        "col_filter": lambda c: c.startswith("sem_"),
        "mi_k": 15,
        "rfe_cv": 2,
        "rfe_min": 2,
    },
    {
        "name": "多模态模型",
        "prefix": "multimodal",
        "col_filter": lambda c: True,
        "mi_k": 30,
        "rfe_cv": 3,
        "rfe_min": 3,
    },
]

print("=" * 70)
print("步骤2-二分类: 训练早期 vs 晚期分类模型（3个版本）")
print("=" * 70)

# ======================== 读取数据 ========================
if not os.path.exists(config.FEATURES_FILE):
    print(f"\n错误: 找不到特征文件 {config.FEATURES_FILE}")
    print("请先运行 'python step1_提取特征.py'")
    exit()

df_all = pd.read_csv(config.FEATURES_FILE)
if "label" in df_all.columns:
    label_col = df_all["label"]
    if label_col.dtype == "object":
        df_all = df_all[df_all["label"] != "未知"].copy()
print(f"\n读取特征表: {len(df_all)}个样品, {len(df_all.columns)}列")

# 二分类标签
df_all["binary_label"] = df_all["label"].map(BINARY_MAP)
unmapped = df_all["binary_label"].isna().sum()
if unmapped > 0:
    print(f"\n[警告] {unmapped} 个样品的标签不在映射表中，已被跳过")
    df_all = df_all[df_all["binary_label"].notna()].copy()

print(f"\n二分类标签分布:")
print(df_all["binary_label"].value_counts())

# 标签编码（全局）
le_global = LabelEncoder()
le_global.fit(df_all["binary_label"])
class_names = [str(c) for c in le_global.classes_]
y_all = le_global.transform(df_all["binary_label"])

all_feature_names = [c for c in df_all.columns if c not in ["sample_name", "label", "binary_label"]]

# ======================== 训练每个模型 ========================
results_summary = []

for mc in MODEL_CONFIGS:
    name = mc["name"]
    prefix = mc["prefix"]
    col_filter = mc["col_filter"]

    print(f"\n{'=' * 70}")
    print(f"--- 训练 {name} (prefix={prefix}) ---")
    print(f"{'=' * 70}")

    # 选择特征子集
    model_cols = [c for c in all_feature_names if col_filter(c)]
    print(f"\n  特征子集: {len(model_cols)} 个")

    # 选择样品子集
    if prefix == "image":
        # 图像模型: 只保留有SEM数据的样品
        sem_cols = [c for c in model_cols]
        has_sem = df_all[sem_cols].sum(axis=1) != 0
        df = df_all[has_sem].copy()
        y = y_all[has_sem]
        print(f"  筛选后有SEM的样品: {len(df)} 个")
    else:
        df = df_all.copy()
        y = y_all

    X_df = df[model_cols]
    feature_names = model_cols
    X = X_df.values

    print(f"\n  样品数: {len(df)}, 特征数: {len(feature_names)}")
    for cls, cls_name in enumerate(class_names):
        count = sum(y == cls)
        print(f"    {cls_name}: {count}个")

    # ======================== 特征选择 ========================
    print("\n  --- 特征选择 ---")

    # 第1步：方差过滤
    var_thresh = VarianceThreshold(threshold=0.01)
    X_var = var_thresh.fit_transform(X)
    var_mask = var_thresh.get_support()
    selected_names_1 = [n for n, k in zip(feature_names, var_mask) if k]
    print(f"    方差过滤: {len(feature_names)} → {len(selected_names_1)}")

    # 第2步：互信息选择
    if len(selected_names_1) > mc["mi_k"]:
        k = mc["mi_k"]
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_mi = mi_selector.fit_transform(X_var, y)
        mi_mask = mi_selector.get_support()
        selected_names_2 = [n for n, k in zip(selected_names_1, mi_mask) if k]
        print(f"    互信息选择: {len(selected_names_1)} → {len(selected_names_2)}")
    else:
        X_mi = X_var
        selected_names_2 = selected_names_1
        print(f"    互信息选择: 已较少，跳过")

    # 第3步：RFECV
    inner_cv = StratifiedKFold(n_splits=min(mc["rfe_cv"], len(df)), shuffle=True, random_state=config.RANDOM_SEED)

    lr_for_rfe = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED, class_weight="balanced")
    rfe = RFECV(estimator=lr_for_rfe, step=1, cv=inner_cv, scoring="accuracy",
                min_features_to_select=mc["rfe_min"])
    rfe.fit(X_mi, y)

    rfe_mask = rfe.support_
    selected_names_final = [n for n, k in zip(selected_names_2, rfe_mask) if k]
    print(f"    RFECV: {len(selected_names_2)} → {len(selected_names_final)}")
    print(f"    最终特征: {', '.join(selected_names_final)}")

    X_selected = X_mi[:, rfe_mask]

    # 特征重要性
    ranking = rfe.ranking_
    importance_order = sorted(zip(selected_names_2, ranking), key=lambda x: x[1])
    print("\n    特征重要性排名:")
    for i, (fname, rank) in enumerate(importance_order[:10]):
        print(f"      {i+1}. {fname} (排名: {rank})")

    # ======================== 数据归一化 ========================
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # ======================== 交叉验证 ========================
    print("\n  --- 交叉验证 ---")
    outer_cv = StratifiedKFold(n_splits=min(config.CV_FOLDS, len(df)), shuffle=True, random_state=config.RANDOM_SEED)

    svm_model = SVC(kernel="rbf", C=config.SVM_C, gamma=config.SVM_GAMMA,
                    class_weight="balanced", random_state=config.RANDOM_SEED, probability=True)

    cv_results = cross_validate(svm_model, X_scaled, y, cv=outer_cv,
                                scoring=["accuracy", "f1_macro", "recall_macro"],
                                return_train_score=True)

    print(f"    准确率:   {cv_results['test_accuracy'].mean():.1%} ± {cv_results['test_accuracy'].std():.1%}")
    print(f"    宏F1:     {cv_results['test_f1_macro'].mean():.1%} ± {cv_results['test_f1_macro'].std():.1%}")
    print(f"    宏召回:   {cv_results['test_recall_macro'].mean():.1%} ± {cv_results['test_recall_macro'].std():.1%}")

    # ======================== 训练最终模型 ========================
    print("\n  --- 训练最终模型 ---")
    svm_model.fit(X_scaled, y)
    y_pred = svm_model.predict(X_scaled)
    train_acc_full = accuracy_score(y, y_pred)
    print(f"  训练集准确率: {train_acc_full:.1%}")

    # ======================== 保存模型 ========================
    print("\n  --- 保存模型 ---")
    joblib.dump(svm_model, os.path.join(config.MODEL_DIR, f"svm_{prefix}_model.joblib"))
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, f"{prefix}_scaler.joblib"))

    selector_info = {
        "var_thresh": var_thresh,
        "mi_selector": mi_selector if len(feature_names) > mc["mi_k"] else None,
        "rfe": rfe,
        "feature_names_all": feature_names,
        "feature_names_after_var": selected_names_1,
        "feature_names_after_mi": selected_names_2,
        "feature_names_final": selected_names_final
    }
    joblib.dump(selector_info, os.path.join(config.MODEL_DIR, f"{prefix}_selector.joblib"))
    joblib.dump(le_global, os.path.join(config.MODEL_DIR, f"{prefix}_label_encoder.joblib"))

    # ======================== 混淆矩阵 ========================
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("预测", fontsize=12)
    ax.set_ylabel("实际", fontsize=12)
    ax.set_title(f"混淆矩阵（{name}）", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close()

    # ======================== PCA散点图 ========================
    if X_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
    else:
        pca = PCA(n_components=2) if X_scaled.shape[1] >= 2 else None
        X_pca = X_scaled if X_scaled.shape[1] >= 1 else np.zeros((X_scaled.shape[0], 2))
        if X_pca.shape[1] == 1:
            X_pca = np.hstack([X_pca, np.zeros((X_pca.shape[0], 1))])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_plot = ["#e74c3c", "#3498db"]
    markers_plot = ["o", "s"]
    for i, cls in enumerate(class_names):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors_plot[i]], marker=markers_plot[i],
                   label=cls, s=100, alpha=0.8, edgecolors="black", linewidths=0.5)
    if pca:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title(f"PCA散点图（{name}）", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, f"{prefix}_pca_scatter.png"), dpi=150)
    plt.close()

    results_summary.append({
        "name": name,
        "prefix": prefix,
        "n_samples": len(df),
        "n_input": len(feature_names),
        "n_selected": len(selected_names_final),
        "cv_acc": cv_results["test_accuracy"].mean(),
        "cv_acc_std": cv_results["test_accuracy"].std(),
        "train_acc": train_acc_full,
    })

# ======================== 打印对比表 ========================
print("\n" + "=" * 70)
print("模型训练对比")
print("=" * 70)
print(f"{'模型':<12} | {'样品数':>6} | {'输入特征':>8} | {'选中特征':>8} | {'CV准确率':>10} | {'训练准确率':>10}")
print("-" * 70)
for r in results_summary:
    print(f"{r['name']:<10} | {r['n_samples']:>6} | {r['n_input']:>8} | {r['n_selected']:>8} | "
          f"{r['cv_acc']:>8.1%} ± {r['cv_acc_std']:.1%} | {r['train_acc']:>8.1%}")

# ======================== 完成 ========================
print("\n" + "=" * 70)
print("全部模型训练完成!")
print("=" * 70)
