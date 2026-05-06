# ============================================================
# app.py —— 龟甲老化分类预测工具（Streamlit 网页版）
# 运行: streamlit run app.py
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
import io
import tempfile
import streamlit as st
import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import config

# ======================== 模型信息 ========================
MODEL_INFO = {
    "spectral": {"label": "光谱模型 (XRD+IR)", "n_train": 14, "desc": "仅使用XRD和IR光谱数据，CV准确率 93.3%"},
    "image": {"label": "图像模型 (SEM)", "n_train": 9, "desc": "仅使用SEM表面/截面图像，CV准确率 40.0%"},
    "multimodal": {"label": "统一多模态模型", "n_train": 14, "desc": "使用XRD+IR+SEM全部数据，CV准确率 80.0%"},
}


# ======================== 编码自动检测 ========================
def auto_detect_encoding(raw_bytes):
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        try:
            return raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass
    try:
        return raw_bytes.decode("gbk")
    except UnicodeDecodeError:
        pass
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    return raw_bytes.decode("latin-1")


def read_csv_auto(file_obj):
    raw = file_obj.getvalue()
    text = auto_detect_encoding(raw)
    return pd.read_csv(io.StringIO(text), header=0)


def stats_skew(data):
    from scipy import stats
    return stats.skew(data)


def stats_kurt(data):
    from scipy import stats
    return stats.kurtosis(data)


# ======================== 特征提取 ========================
def extract_xrd(df_raw):
    angles = df_raw.iloc[:, 0].values.astype(float)
    intensities = df_raw.iloc[:, 1].values.astype(float)
    features = {}
    features["xrd_mean_intensity"] = np.mean(intensities)
    features["xrd_max_intensity"] = np.max(intensities)
    features["xrd_std_intensity"] = np.std(intensities)
    features["xrd_total_area"] = np.trapz(intensities, angles)
    features["xrd_skewness"] = stats_skew(intensities) if len(intensities) > 2 else 0.0
    features["xrd_kurtosis"] = stats_kurt(intensities) if len(intensities) > 3 else 0.0

    from scipy.signal import find_peaks
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

    from scipy.signal import find_peaks
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
    import cv2
    from scipy.ndimage import label as nd_label
    from scipy import stats
    from skimage.feature import graycomatrix, graycoprops
    from skimage.feature import local_binary_pattern
    from skimage.filters import threshold_otsu as otsu_threshold

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


def extract_sem_from_files(uploaded_files, prefix):
    """从上传的SEM图片提取特征，多张图片取平均。prefix: 'sem_s_' 或 'sem_c_'"""
    if not uploaded_files:
        return {}
    all_feats = []
    for uf in uploaded_files:
        ext = os.path.splitext(uf.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        feat = extract_sem_single(tmp_path)
        os.unlink(tmp_path)
        if feat is not None:
            all_feats.append(feat)
    if not all_feats:
        return {}
    keys = list(all_feats[0].keys())
    return {f"{prefix}{k}": np.mean([f[k] for f in all_feats]) for k in keys}


def apply_feature_selection(features_dict, selector_info, feature_names_all):
    X = np.array([features_dict.get(name, 0.0) for name in feature_names_all])
    X_var = selector_info["var_thresh"].transform(X.reshape(1, -1))
    if selector_info["mi_selector"] is not None:
        X_mi = selector_info["mi_selector"].transform(X_var)
    else:
        X_mi = X_var
    X_final = selector_info["rfe"].transform(X_mi)
    return X_final


# ======================== 页面 ========================
st.set_page_config(page_title="龟甲老化分类", page_icon="🐢", layout="centered")
st.title("龟甲老化程度分类")

# ======================== 模型选择器 ========================
model_type = st.radio(
    "选择预测模型",
    options=["spectral", "image", "multimodal"],
    format_func=lambda x: MODEL_INFO[x]["label"],
    horizontal=True,
)
st.caption(MODEL_INFO[model_type]["desc"])


# ======================== 加载模型 ========================
@st.cache_resource
def load_model(prefix):
    model = joblib.load(os.path.join(config.MODEL_DIR, f"svm_{prefix}_model.joblib"))
    scaler = joblib.load(os.path.join(config.MODEL_DIR, f"{prefix}_scaler.joblib"))
    selector = joblib.load(os.path.join(config.MODEL_DIR, f"{prefix}_selector.joblib"))
    le = joblib.load(os.path.join(config.MODEL_DIR, f"{prefix}_label_encoder.joblib"))
    return model, scaler, selector, le


# ======================== 条件上传区 ========================
st.divider()
st.subheader("上传数据")

xrd_file = None
ir_file = None
sem_surface_files = None
sem_cross_files = None

if model_type != "image":
    col1, col2 = st.columns(2)
    with col1:
        xrd_file = st.file_uploader("XRD 文件 (.csv)", type=["csv"], key="xrd")
    with col2:
        ir_file = st.file_uploader("IR 文件 (.csv)", type=["csv", "CSV"], key="ir")

if model_type != "spectral":
    st.markdown("**SEM 图像上传**")
    col3, col4 = st.columns(2)
    with col3:
        sem_surface_files = st.file_uploader(
            "表面图像 (可多选)",
            type=["jpg", "jpeg", "tif", "tiff", "png", "bmp"],
            accept_multiple_files=True,
            key="sem_surface",
        )
    with col4:
        sem_cross_files = st.file_uploader(
            "截面图像 (可多选)",
            type=["jpg", "jpeg", "tif", "tiff", "png", "bmp"],
            accept_multiple_files=True,
            key="sem_cross",
        )

# ======================== 数据预览 ========================
if model_type != "spectral" and sem_surface_files:
    st.markdown("**已上传的表面图像:**")
    cols = st.columns(min(len(sem_surface_files), 4))
    for i, uf in enumerate(sem_surface_files):
        cols[i % 4].image(uf, caption=uf.name, width=150)

if model_type != "spectral" and sem_cross_files:
    st.markdown("**已上传的截面图像:**")
    cols = st.columns(min(len(sem_cross_files), 4))
    for i, uf in enumerate(sem_cross_files):
        cols[i % 4].image(uf, caption=uf.name, width=150)

# ======================== 预测 ========================
if st.button("开始预测", type="primary", use_container_width=True):
    # 输入验证
    has_spectral_input = (xrd_file is not None) or (ir_file is not None)
    has_image_input = (sem_surface_files is not None and len(sem_surface_files) > 0) or \
                      (sem_cross_files is not None and len(sem_cross_files) > 0)

    if model_type == "spectral" and not has_spectral_input:
        st.error("请上传 XRD 或 IR 文件")
        st.stop()
    elif model_type == "image" and not has_image_input:
        st.error("请上传 SEM 表面或截面图像")
        st.stop()
    elif model_type == "multimodal" and not has_spectral_input and not has_image_input:
        st.error("请至少上传一种数据源")
        st.stop()

    # 加载模型
    model, scaler, selector, le = load_model(model_type)
    feature_names_all = selector["feature_names_all"]

    # 提取特征
    all_feat = {}
    data_sources = []

    if xrd_file:
        try:
            df_xrd = read_csv_auto(xrd_file)
            all_feat.update(extract_xrd(df_xrd))
            data_sources.append("XRD")
        except Exception as e:
            st.error(f"XRD 文件解析失败: {e}")

    if ir_file:
        try:
            df_ir = read_csv_auto(ir_file)
            all_feat.update(extract_ir(df_ir))
            data_sources.append("IR")
        except Exception as e:
            st.error(f"IR 文件解析失败: {e}")

    if sem_surface_files:
        sem_s_feat = extract_sem_from_files(sem_surface_files, "sem_s_")
        if sem_s_feat:
            all_feat.update(sem_s_feat)
            data_sources.append("SEM表面")

    if sem_cross_files:
        sem_c_feat = extract_sem_from_files(sem_cross_files, "sem_c_")
        if sem_c_feat:
            all_feat.update(sem_c_feat)
            data_sources.append("SEM截面")

    if not all_feat:
        st.error("未提取到任何有效特征，请检查上传的文件。")
        st.stop()

    # CSV文件预览
    if model_type != "image":
        st.divider()
        st.subheader("数据预览")
        if xrd_file:
            xrd_file.seek(0)
            df_xrd_preview = read_csv_auto(xrd_file)
            st.markdown(f"**XRD 文件:** {len(df_xrd_preview)} 行 × {len(df_xrd_preview.columns)} 列")
            st.dataframe(df_xrd_preview.head(5), use_container_width=True)
        if ir_file:
            ir_file.seek(0)
            df_ir_preview = read_csv_auto(ir_file)
            st.markdown(f"**IR 文件:** {len(df_ir_preview)} 行 × {len(df_ir_preview.columns)} 列")
            st.dataframe(df_ir_preview.head(5), use_container_width=True)

    # 预测
    X_sel = apply_feature_selection(all_feat, selector, feature_names_all)
    X_sc = scaler.transform(X_sel)
    pred = model.predict(X_sc)[0]
    pred_name = le.inverse_transform([pred])[0]
    probs = model.predict_proba(X_sc)[0]
    class_idx = list(model.classes_).index(pred)
    confidence = probs[class_idx] * 100

    # 显示结果
    st.divider()
    st.subheader("预测结果")

    if pred_name == "早期老化":
        st.success(f"预测结果：**{pred_name}**（置信度 {confidence:.1f}%）")
        st.info("对应老化级别：1级轻度 或 2级中度")
    else:
        st.warning(f"预测结果：**{pred_name}**（置信度 {confidence:.1f}%）")
        st.info("对应老化级别：3级重度 或 4级严重")

    # 概率条
    st.markdown("**各类别概率：**")
    early_prob = 0
    late_prob = 0
    for i, cls in enumerate(model.classes_):
        name = le.inverse_transform([cls])[0]
        prob = probs[i] * 100
        if name == "早期老化":
            early_prob = prob
        else:
            late_prob = prob

    st.progress(early_prob / 100, text=f"早期老化: {early_prob:.1f}%")
    st.progress(late_prob / 100, text=f"晚期老化: {late_prob:.1f}%")

    # 可靠性评估
    X_train = pd.read_csv(os.path.join(PROJECT_DIR, "features.csv"))
    train_ranges = {}
    for f in selector["feature_names_final"]:
        vals = X_train[f]
        train_ranges[f] = {"min": float(vals.min()), "max": float(vals.max())}

    out_of_range_count = 0
    ood_features = []
    for fname, rng in train_ranges.items():
        val = all_feat.get(fname, 0.0)
        if val < rng["min"] or val > rng["max"]:
            out_of_range_count += 1
            ood_features.append(fname)

    total_final_feats = len(selector["feature_names_final"])
    has_xrd = "XRD" in data_sources
    has_ir = "IR" in data_sources
    has_sem_s = "SEM表面" in data_sources
    has_sem_c = "SEM截面" in data_sources

    if model_type == "multimodal":
        modalities = []
        missing = []
        if has_xrd: modalities.append("XRD")
        else: missing.append("XRD")
        if has_ir: modalities.append("IR")
        else: missing.append("IR")
        if has_sem_s or has_sem_c: modalities.append("SEM")
        else: missing.append("SEM")

        if out_of_range_count > 0:
            st.warning(f"**可靠性较低** — {out_of_range_count}/{total_final_feats} 个特征超出训练范围。\n"
                       f"`{', '.join(ood_features)}`")
        elif len(missing) > 0:
            st.info(f"**可靠性中等** — 缺少 {', '.join(missing)} 数据，模型使用默认值(0)填充。")
        else:
            st.success("**可靠性较高** — 所有特征均在训练集范围内。")

        st.caption(f"数据模态: {' + '.join(modalities)} | 使用数据源: {', '.join(data_sources)}")

    elif model_type == "spectral":
        if out_of_range_count > 0:
            st.warning(f"**可靠性较低** — {out_of_range_count}/{total_final_feats} 个特征超出训练范围。\n"
                       f"`{', '.join(ood_features)}`")
        elif not has_xrd or not has_ir:
            missing = []
            if not has_xrd: missing.append("XRD")
            if not has_ir: missing.append("IR")
            st.info(f"**可靠性中等** — 缺少 {', '.join(missing)} 数据。")
        else:
            st.success("**可靠性较高** — 所有特征均在训练集范围内。")

    elif model_type == "image":
        if out_of_range_count > 0:
            st.warning(f"**可靠性较低** — {out_of_range_count}/{total_final_feats} 个特征超出训练范围。\n"
                       f"`{', '.join(ood_features)}`")
        elif not has_sem_s or not has_sem_c:
            missing = []
            if not has_sem_s: missing.append("表面")
            if not has_sem_c: missing.append("截面")
            st.info(f"**可靠性中等** — 缺少 SEM{', '.join(missing)} 图像。")
        else:
            st.success("**可靠性较高** — 所有特征均在训练集范围内。")

    st.caption(f"模型: {MODEL_INFO[model_type]['label']} | 训练样品: {MODEL_INFO[model_type]['n_train']}个")

    # 调试面板
    with st.expander("调试信息 — 查看提取的特征"):
        st.markdown("**提取到的全部特征:**")
        feat_df = pd.DataFrame(sorted(all_feat.items()), columns=["特征名", "值"])
        st.dataframe(feat_df, use_container_width=True, height=300)

        st.divider()
        st.markdown("**模型使用的关键特征 vs 训练集范围:**")
        for fname in selector["feature_names_final"]:
            val = all_feat.get(fname, 0.0)
            mn = train_ranges[fname]["min"]
            mx = train_ranges[fname]["max"]
            st.markdown(f"**{fname}** = `{val:.4f}`")
            if val < mn:
                zone = f"→ ⚠️ **低于**训练范围 (min={mn:.2f})"
            elif val > mx:
                zone = f"→ ⚠️ **高于**训练范围 (max={mx:.2f})"
            else:
                zone = f"→ 在训练范围内 ({mn:.2f}–{mx:.2f})"
            st.caption(f"  {zone}")

# 底部说明
st.divider()
st.caption("使用方法：1. 选择模型 → 2. 上传对应数据 → 3. 点击'开始预测' → 4. 查看结果")
st.caption("注：建议先用'光谱模型'做基础判断，有SEM数据时可切换'多模态模型'交叉验证。")
