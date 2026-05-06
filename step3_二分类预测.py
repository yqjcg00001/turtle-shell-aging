# ============================================================
# step3_二分类预测.py —— 二分类版本的预测（早期 vs 晚期）
# ============================================================
# 用法:
#   python step3_二分类预测.py --xrd data/xrd/xxx_xrd.csv --ir data/ir/xxx_ir.csv
#   python step3_二分类预测.py --xrd data/xrd/xxx_xrd.csv  # 只有XRD也行
#   python step3_二分类预测.py --ir data/ir/xxx_ir.csv     # 只有IR也行
# ============================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import label as nd_label
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu as otsu_threshold
import config


# ======================== 特征提取函数 ========================
def extract_xrd_features_from_file(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, header=0)
    cols = df.columns.tolist()
    angles = df[cols[0]].values
    intensities = df[cols[1]].values
    features = {}
    features["xrd_mean_intensity"] = np.mean(intensities)
    features["xrd_max_intensity"] = np.max(intensities)
    features["xrd_std_intensity"] = np.std(intensities)
    features["xrd_total_area"] = np.trapz(intensities, angles)
    if len(intensities) > 2:
        features["xrd_skewness"] = stats.skew(intensities)
    else:
        features["xrd_skewness"] = 0.0
    if len(intensities) > 3:
        features["xrd_kurtosis"] = stats.kurtosis(intensities)
    else:
        features["xrd_kurtosis"] = 0.0
    peak_indices, peak_props = find_peaks(intensities, prominence=config.XRD_PEAK_PROMINENCE, distance=config.XRD_PEAK_DISTANCE)
    features["xrd_peak_count"] = len(peak_indices)
    if len(peak_indices) > 0:
        peak_heights = peak_props["prominences"]
        sorted_idx = np.argsort(peak_heights)[::-1]
        top_n = min(3, len(peak_indices))
        for i in range(top_n):
            idx = sorted_idx[i]
            features[f"xrd_peak{i+1}_position"] = angles[peak_indices[idx]]
            features[f"xrd_peak{i+1}_height"] = peak_heights[idx]
        for i in range(top_n, 3):
            features[f"xrd_peak{i+1}_position"] = 0.0
            features[f"xrd_peak{i+1}_height"] = 0.0
    else:
        for i in range(1, 4):
            features[f"xrd_peak{i}_position"] = 0.0
            features[f"xrd_peak{i}_height"] = 0.0
    crystalline_area = 0
    for region in config.XRD_CRYSTALLINE_REGIONS:
        mask = (angles >= region[0]) & (angles <= region[1])
        if np.any(mask):
            crystalline_area += np.trapz(intensities[mask], angles[mask])
    amorphous_mask = (angles >= config.XRD_AMORPHOUS_REGION[0]) & (angles <= config.XRD_AMORPHOUS_REGION[1])
    amorphous_area = np.trapz(intensities[amorphous_mask], angles[amorphous_mask]) if np.any(amorphous_mask) else 0.0
    total = crystalline_area + amorphous_area
    features["xrd_crystallinity_index"] = crystalline_area / total if total > 0 else 0.0
    if len(peak_indices) > 0:
        main_peak_idx = peak_indices[sorted_idx[0]]
        main_peak_pos = angles[main_peak_idx]
        main_peak_height = intensities[main_peak_idx]
        half_max = main_peak_height / 2.0
        left_idx = main_peak_idx
        while left_idx > 0 and intensities[left_idx] > half_max:
            left_idx -= 1
        right_idx = main_peak_idx
        while right_idx < len(intensities) - 1 and intensities[right_idx] > half_max:
            right_idx += 1
        features["xrd_fwhm"] = angles[right_idx] - angles[left_idx]
        fwhm_rad = np.deg2rad(features["xrd_fwhm"])
        features["xrd_crystallite_size"] = 0.9 * 1.5406 / (fwhm_rad * np.cos(np.deg2rad(main_peak_pos))) if fwhm_rad > 0 else 0.0
    else:
        features["xrd_fwhm"] = 0.0
        features["xrd_crystallite_size"] = 0.0
    return features


def extract_ir_features_from_file(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, header=0)
    cols = df.columns.tolist()
    wavenumbers = df[cols[0]].values
    absorbance = np.abs(df[cols[1]].values)
    features = {}
    features["ir_mean_absorbance"] = np.mean(absorbance)
    features["ir_max_absorbance"] = np.max(absorbance)
    features["ir_std_absorbance"] = np.std(absorbance)
    features["ir_total_area"] = np.trapz(absorbance, wavenumbers)
    peak_indices, peak_props = find_peaks(absorbance, prominence=0.05, distance=10)
    features["ir_peak_count"] = len(peak_indices)
    if len(peak_indices) > 0:
        peak_heights = peak_props["prominences"]
        sorted_idx = np.argsort(peak_heights)[::-1]
        top_n = min(5, len(peak_indices))
        for i in range(top_n):
            idx = sorted_idx[i]
            features[f"ir_peak{i+1}_position"] = wavenumbers[peak_indices[idx]]
            features[f"ir_peak{i+1}_height"] = peak_heights[idx]
        for i in range(top_n, 5):
            features[f"ir_peak{i+1}_position"] = 0.0
            features[f"ir_peak{i+1}_height"] = 0.0
    else:
        for i in range(1, 6):
            features[f"ir_peak{i}_position"] = 0.0
            features[f"ir_peak{i}_height"] = 0.0
    for group_name, (wn_min, wn_max) in config.IR_FUNCTIONAL_GROUPS.items():
        mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)
        if np.any(mask):
            features[f"ir_{group_name}_max"] = np.max(absorbance[mask])
            features[f"ir_{group_name}_area"] = np.trapz(absorbance[mask], wavenumbers[mask])
            features[f"ir_{group_name}_mean"] = np.mean(absorbance[mask])
        else:
            features[f"ir_{group_name}_max"] = 0.0
            features[f"ir_{group_name}_area"] = 0.0
            features[f"ir_{group_name}_mean"] = 0.0
    if "ir_PO4磷酸根_area" in features and "ir_CO3碳酸根_area" in features:
        features["ir_ratio_PO4_CO3"] = features["ir_PO4磷酸根_area"] / features["ir_CO3碳酸根_area"] if features["ir_CO3碳酸根_area"] > 0 else 0.0
    else:
        features["ir_ratio_PO4_CO3"] = 0.0
    if "ir_酰胺I_area" in features and "ir_酰胺II_area" in features:
        features["ir_ratio_amide1_amide2"] = features["ir_酰胺I_area"] / features["ir_酰胺II_area"] if features["ir_酰胺II_area"] > 0 else 0.0
    else:
        features["ir_ratio_amide1_amide2"] = 0.0
    return features


def extract_sem_single(filepath):
    if not os.path.exists(filepath):
        return None
    image = cv2.imread(filepath)
    if image is None:
        return None
    image = cv2.resize(image, config.SEM_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}
    features["gray_mean"] = np.mean(gray)
    features["gray_std"] = np.std(gray)
    features["gray_median"] = np.median(gray)
    features["gray_min"] = np.min(gray)
    features["gray_max"] = np.max(gray)
    features["gray_skewness"] = stats.skew(gray.flatten())
    features["gray_kurtosis"] = stats.kurtosis(gray.flatten())
    glcm = graycomatrix(gray, distances=[config.GLCM_DISTANCE], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=config.GLCM_LEVELS, symmetric=True, normed=True)
    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        features[f"glcm_{prop_name}"] = np.mean(graycoprops(glcm, prop_name))
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, "uniform")
    n_bins = int(n_points + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    for i in range(n_bins):
        features[f"lbp_{i}"] = hist[i]
    threshold = otsu_threshold(gray)
    binary = gray > threshold
    labeled, num_features = nd_label(binary.astype(int))
    features["particle_count"] = num_features
    if num_features > 0:
        areas = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        features["morph_mean_area"] = np.mean(areas)
        features["morph_std_area"] = np.std(areas)
        features["morph_max_area"] = np.max(areas)
        features["morph_area_fraction"] = np.sum(binary) / gray.size
        features["morph_area_cv"] = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0.0
    else:
        for k in ["morph_mean_area", "morph_std_area", "morph_max_area", "morph_area_fraction", "morph_area_cv"]:
            features[k] = 0.0
    return features


def apply_feature_selection(features_dict, selector_info, feature_names_all):
    X = np.array([features_dict.get(name, 0.0) for name in feature_names_all])
    X_var = selector_info["var_thresh"].transform(X.reshape(1, -1))
    if selector_info["mi_selector"] is not None:
        X_mi = selector_info["mi_selector"].transform(X_var)
    else:
        X_mi = X_var
    X_final = selector_info["rfe"].transform(X_mi)
    return X_final


# ======================== 主程序 ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="二分类预测：早期老化 vs 晚期老化")
    parser.add_argument("--xrd", type=str, default=None, help="XRD的CSV文件路径")
    parser.add_argument("--ir", type=str, default=None, help="IR的CSV文件路径")
    parser.add_argument("--sem_surface", type=str, nargs="*", default=None, help="SEM表面图片")
    parser.add_argument("--sem_cross", type=str, nargs="*", default=None, help="SEM截面图片")
    args = parser.parse_args()

    if all(v is None for v in [args.xrd, args.ir, args.sem_surface, args.sem_cross]):
        print("错误: 至少要提供一个数据源！")
        print("用法: python step3_二分类预测.py --xrd data/xrd/xxx.csv --ir data/ir/xxx.csv")
        exit()

    print("=" * 60)
    print("二分类预测: 早期老化 vs 晚期老化")
    print("=" * 60)

    # 加载模型
    model = joblib.load(os.path.join(config.MODEL_DIR, "svm_binary_model.joblib"))
    scaler = joblib.load(os.path.join(config.MODEL_DIR, "binary_scaler.joblib"))
    selector_info = joblib.load(os.path.join(config.MODEL_DIR, "binary_selector.joblib"))
    le = joblib.load(os.path.join(config.MODEL_DIR, "binary_label_encoder.joblib"))
    feature_names_all = selector_info["feature_names_all"]

    # 提取特征
    print("\n提取特征:")
    all_feat = {}
    if args.xrd:
        xrd = extract_xrd_features_from_file(args.xrd)
        if xrd:
            all_feat.update(xrd)
            print(f"  XRD: {len(xrd)} 个特征")
    if args.ir:
        ir = extract_ir_features_from_file(args.ir)
        if ir:
            all_feat.update(ir)
            print(f"  IR:  {len(ir)} 个特征")
    if args.sem_surface:
        feats = [extract_sem_single(f) for f in args.sem_surface if extract_sem_single(f)]
        if feats:
            sem_s = {f"sem_s_{k}": np.mean([f[k] for f in feats]) for k in feats[0].keys()}
            all_feat.update(sem_s)
            print(f"  SEM表面: {len(sem_s)} 个特征")
    if args.sem_cross:
        feats = [extract_sem_single(f) for f in args.sem_cross if extract_sem_single(f)]
        if feats:
            sem_c = {f"sem_c_{k}": np.mean([f[k] for f in feats]) for k in feats[0].keys()}
            all_feat.update(sem_c)
            print(f"  SEM截面: {len(sem_c)} 个特征")

    X_sel = apply_feature_selection(all_feat, selector_info, feature_names_all)
    X_sc = scaler.transform(X_sel)

    pred = model.predict(X_sc)[0]
    pred_name = le.inverse_transform([pred])[0]
    probs = model.predict_proba(X_sc)[0]
    class_idx = list(model.classes_).index(pred)
    confidence = probs[class_idx] * 100

    print("\n" + "=" * 60)
    print("预测结果:")
    print("=" * 60)
    print(f"  预测: {pred_name}")
    print(f"  置信度: {confidence:.1f}%")
    print(f"\n  各类别概率:")
    for i, cls in enumerate(model.classes_):
        name = le.inverse_transform([cls])[0]
        print(f"    {name}: {probs[i]*100:.1f}%")

    if confidence >= 80:
        print(f"\n  [OK] 模型非常有把握，该样品很可能是 '{pred_name}'")
    elif confidence >= 60:
        print(f"\n  [注意] 模型比较有信心，建议结合专业知识进一步确认")
    else:
        print(f"\n  [警告] 模型不太确定，建议结合多种手段综合判断")

    print(f"\n{'=' * 60}")
    print("预测完成!")
