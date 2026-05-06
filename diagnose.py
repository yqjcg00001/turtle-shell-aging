# ============================================================
# diagnose.py —— 诊断特征提取和预测是否一致
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
import io
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from scipy.signal import find_peaks
from scipy import stats

# 加载模型
selector = joblib.load("models/binary_selector.joblib")
scaler = joblib.load("models/binary_scaler.joblib")
model = joblib.load("models/svm_binary_model.joblib")
le = joblib.load("models/binary_label_encoder.joblib")
feature_names_all = selector["feature_names_all"]

# 读取训练时的特征
train_df = pd.read_csv("features.csv")

def auto_detect_encoding(raw_bytes):
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return raw_bytes.decode("utf-8-sig")
    try:
        return raw_bytes.decode("gbk")
    except:
        pass
    try:
        return raw_bytes.decode("utf-8")
    except:
        pass
    return raw_bytes.decode("latin-1")

def read_csv_auto(path):
    with open(path, "rb") as f:
        raw = f.read()
    text = auto_detect_encoding(raw)
    return pd.read_csv(io.StringIO(text), header=0)

def extract_xrd(df_raw):
    angles = df_raw.iloc[:, 0].values.astype(float)
    intensities = df_raw.iloc[:, 1].values.astype(float)
    features = {}
    features["xrd_mean_intensity"] = np.mean(intensities)
    features["xrd_max_intensity"] = np.max(intensities)
    features["xrd_std_intensity"] = np.std(intensities)
    features["xrd_total_area"] = np.trapz(intensities, angles)
    features["xrd_skewness"] = stats.skew(intensities) if len(intensities) > 2 else 0.0
    features["xrd_kurtosis"] = stats.kurtosis(intensities) if len(intensities) > 3 else 0.0
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

def extract_ir(df_raw):
    wavenumbers = df_raw.iloc[:, 0].values.astype(float)
    absorbance = np.abs(df_raw.iloc[:, 1].values.astype(float))
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

def predict_from_features(all_feat):
    X = np.array([all_feat.get(n, 0.0) for n in feature_names_all])
    X_var = selector["var_thresh"].transform(X.reshape(1, -1))
    if selector["mi_selector"] is not None:
        X_mi = selector["mi_selector"].transform(X_var)
    else:
        X_mi = X_var
    X_final = selector["rfe"].transform(X_mi)
    X_scaled = scaler.transform(X_final)
    pred = model.predict(X_scaled)[0]
    pred_name = le.inverse_transform([pred])[0]
    probs = model.predict_proba(X_scaled)[0]
    class_idx = list(model.classes_).index(pred)
    conf = probs[class_idx] * 100
    return pred_name, conf, X_final.flatten()

# ======================== 对比训练时特征 vs 重新提取 ========================
final_feats = selector["feature_names_final"]

print("=" * 70)
print("对比：训练时 features.csv vs 从原始文件重新提取")
print("=" * 70)

for sample in ["919_003", "919_009", "000", "919_001", "919_005"]:
    train_row = train_df[train_df["sample_name"] == sample]
    if train_row.empty:
        print(f"\n{sample}: 训练数据中不存在")
        continue

    label = train_row.iloc[0]["label"]
    # 训练时用于分类的标签
    binary_map = {1: "早期老化", 2: "早期老化", 3: "晚期老化", 4: "晚期老化"}
    true_binary = binary_map.get(label, "?")

    print(f"\n--- {sample} (原始标签={label}, 二分类={true_binary}) ---")

    # 训练时的特征值
    print("  训练时特征值 (features.csv):")
    for feat in final_feats:
        val = train_row.iloc[0][feat]
        print(f"    {feat}: {val:.4f}")

    # 从原始文件重新提取
    all_feat = {}
    xrd_path = f"data/xrd/{sample}_xrd.csv"
    ir_path = f"data/ir/{sample}_ir.CSV"
    if os.path.exists(xrd_path):
        df_xrd = read_csv_auto(xrd_path)
        all_feat.update(extract_xrd(df_xrd))
    if os.path.exists(ir_path):
        df_ir = read_csv_auto(ir_path)
        all_feat.update(extract_ir(df_ir))

    print("  重新提取特征:")
    for feat in final_feats:
        val = all_feat.get(feat, "MISSING")
        if isinstance(val, float):
            print(f"    {feat}: {val:.4f}")
        else:
            print(f"    {feat}: {val}")

    # 预测
    pred_name, conf, X_final = predict_from_features(all_feat)
    print(f"  → 预测: {pred_name} (置信度 {conf:.1f}%)")

    # 对比关键特征
    print("  关键特征对比:")
    for feat in final_feats:
        train_val = train_row.iloc[0][feat]
        new_val = all_feat.get(feat, "MISSING")
        if isinstance(new_val, (int, float)):
            diff = abs(train_val - new_val)
            match = "OK" if diff < 0.001 else f"DIFF={diff:.4f}"
        else:
            match = "MISSING"
        print(f"    {feat}: train={train_val:.4f} vs new={new_val} -> {match}")

print("\n" + "=" * 70)
print("全部14个样品的预测结果:")
print("=" * 70)
for _, row in train_df.iterrows():
    sample = row["sample_name"]
    label = row["label"]
    binary_map = {1: "早期老化", 2: "早期老化", 3: "晚期老化", 4: "晚期老化"}
    true_binary = binary_map.get(label, "?")

    all_feat = {}
    xrd_path = f"data/xrd/{sample}_xrd.csv"
    ir_path = f"data/ir/{sample}_ir.CSV"
    if os.path.exists(xrd_path):
        df_xrd = read_csv_auto(xrd_path)
        all_feat.update(extract_xrd(df_xrd))
    if os.path.exists(ir_path):
        df_ir = read_csv_auto(ir_path)
        all_feat.update(extract_ir(df_ir))

    pred_name, conf, _ = predict_from_features(all_feat)
    correct = "OK" if pred_name == true_binary else "WRONG"
    print(f"  {sample}: 实际={true_binary} (label={label}) -> 预测={pred_name} ({conf:.1f}%) [{correct}]")
