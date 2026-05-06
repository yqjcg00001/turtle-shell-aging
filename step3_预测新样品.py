# ============================================================
# step3_预测新样品.py —— 来了新样品，一键预测老化级别
# ============================================================
# 这一步做什么：
#   1. 你提供了一个新龟甲样品的数据（XRD/IR/SEM，可以只有一种）
#   2. 脚本自动提取和训练时一样的特征
#   3. 用训练好的模型预测这个样品属于哪个老化级别
#   4. 输出预测结果、置信度（把握有多大）、各类别概率
#
# 你需要做的：
#   1. 确保已经运行过 step2_训练模型.py（有训练好的模型）
#   2. 准备好新样品的数据文件（至少一种）
#   3. 运行命令（把路径换成你实际的文件路径）:
#
#      三种都有：
#      python step3_预测新样品.py --xrd data/xrd/GH001_xrd.csv --ir data/ir/GH001_ir.csv --sem_surface data/sem/GH001_surface_5000x.jpg --sem_cross data/sem/GH001_cross_10000x.jpg
#
#      只有IR：
#      python step3_预测新样品.py --ir data/ir/GH001_ir.csv
#
#      只有IR+SEM表面：
#      python step3_预测新样品.py --ir data/ir/GH001_ir.csv --sem_surface data/sem/GH001_surface_5000x.jpg
# ============================================================

# ======================== 导入需要的库 ========================
import os                                     # 操作系统接口，处理文件路径
import sys                                    # 系统模块，用来读取命令行参数
import argparse                               # 命令行参数解析工具，帮你处理 --xrd 这样的参数
import numpy as np                            # 数值计算库
import pandas as pd                           # 表格处理库
import joblib                                 # 模型加载工具
import cv2                                    # OpenCV图像处理库
from scipy.signal import find_peaks           # 找峰函数
from scipy.ndimage import label as nd_label   # 连通区域标记
from scipy import stats                       # 统计函数
from skimage.feature import graycomatrix, graycoprops  # GLCM纹理分析
from skimage.feature import local_binary_pattern         # LBP纹理分析
from skimage.filters import threshold_otsu as otsu_threshold  # Otsu自动阈值
import config                                 # 导入配置文件


# ======================== 特征提取函数（和step1中一样） ========================
# 注意：这里必须使用和 step1_提取特征.py 中完全一样的特征提取方法！
# 否则训练和预测时特征不一致，模型就无法正常工作

def extract_xrd_features_from_file(filepath):
    """从XRD CSV文件提取特征（和step1中的逻辑完全一致）"""
    if not os.path.exists(filepath):           # 如果文件不存在
        print(f"  警告: XRD文件不存在: {filepath}")  # 打印警告
        return None                             # 返回None表示缺失

    df = pd.read_csv(filepath, header=0)        # 读取CSV文件
    cols = df.columns.tolist()                  # 获取列名
    angles = df[cols[0]].values                 # 第一列是角度
    intensities = df[cols[1]].values            # 第二列是强度

    features = {}                               # 空字典

    # 基本统计量
    features["xrd_mean_intensity"] = np.mean(intensities)  # 平均强度
    features["xrd_max_intensity"] = np.max(intensities)    # 最大强度
    features["xrd_std_intensity"] = np.std(intensities)    # 强度标准差
    features["xrd_total_area"] = np.trapz(intensities, angles)  # 谱图总面积

    if len(intensities) > 2:                   # 数据足够才计算偏度
        features["xrd_skewness"] = stats.skew(intensities)  # 偏度
    else:
        features["xrd_skewness"] = 0.0

    if len(intensities) > 3:                   # 数据足够才计算峰度
        features["xrd_kurtosis"] = stats.kurtosis(intensities)  # 峰度
    else:
        features["xrd_kurtosis"] = 0.0

    # 找峰
    peak_indices, peak_props = find_peaks(     # 调用找峰函数
        intensities,                           # 强度数据
        prominence=config.XRD_PEAK_PROMINENCE, # 最低显著性
        distance=config.XRD_PEAK_DISTANCE      # 最小距离
    )
    features["xrd_peak_count"] = len(peak_indices)  # 峰的数量

    if len(peak_indices) > 0:                  # 如果找到了峰
        peak_heights = peak_props["prominences"]  # 每个峰的显著性
        sorted_idx = np.argsort(peak_heights)[::-1]  # 从高到低排序
        top_n = min(3, len(peak_indices))      # 最多取3个

        for i in range(top_n):                 # 遍历最强的峰
            idx = sorted_idx[i]
            features[f"xrd_peak{i+1}_position"] = angles[peak_indices[idx]]  # 峰的角度位置
            features[f"xrd_peak{i+1}_height"] = peak_heights[idx]  # 峰的显著性

        for i in range(top_n, 3):              # 补齐到3个
            features[f"xrd_peak{i+1}_position"] = 0.0
            features[f"xrd_peak{i+1}_height"] = 0.0
    else:
        features["xrd_peak1_position"] = 0.0   # 没有峰就填0
        features["xrd_peak1_height"] = 0.0
        features["xrd_peak2_position"] = 0.0
        features["xrd_peak2_height"] = 0.0
        features["xrd_peak3_position"] = 0.0
        features["xrd_peak3_height"] = 0.0

    # 结晶度指数
    crystalline_area = 0                       # 结晶区域面积初始值
    for region in config.XRD_CRYSTALLINE_REGIONS:  # 遍历每个结晶区域
        mask = (angles >= region[0]) & (angles <= region[1])  # 找这个范围内的数据点
        if np.any(mask):                       # 如果有数据点
            crystalline_area += np.trapz(intensities[mask], angles[mask])  # 积分求面积

    amorphous_mask = (angles >= config.XRD_AMORPHOUS_REGION[0]) & (angles <= config.XRD_AMORPHOUS_REGION[1])
    if np.any(amorphous_mask):
        amorphous_area = np.trapz(intensities[amorphous_mask], angles[amorphous_mask])
    else:
        amorphous_area = 0.0

    total = crystalline_area + amorphous_area  # 总面积
    if total > 0:                              # 避免除以0
        features["xrd_crystallinity_index"] = crystalline_area / total  # 结晶度指数
    else:
        features["xrd_crystallinity_index"] = 0.0

    # FWHM和晶粒尺寸
    if len(peak_indices) > 0:
        main_peak_idx = peak_indices[sorted_idx[0]]  # 最强峰的索引
        main_peak_pos = angles[main_peak_idx]   # 最强峰的角度
        main_peak_height = intensities[main_peak_idx]  # 最强峰的高度
        half_max = main_peak_height / 2.0       # 半高值

        left_idx = main_peak_idx                # 向左搜索
        while left_idx > 0 and intensities[left_idx] > half_max:
            left_idx -= 1
        right_idx = main_peak_idx               # 向右搜索
        while right_idx < len(intensities) - 1 and intensities[right_idx] > half_max:
            right_idx += 1

        features["xrd_fwhm"] = angles[right_idx] - angles[left_idx]  # 半高宽

        fwhm_rad = np.deg2rad(features["xrd_fwhm"])  # 角度转弧度
        if fwhm_rad > 0:
            features["xrd_crystallite_size"] = 0.9 * 1.5406 / (fwhm_rad * np.cos(np.deg2rad(main_peak_pos)))  # Scherrer公式
        else:
            features["xrd_crystallite_size"] = 0.0
    else:
        features["xrd_fwhm"] = 0.0
        features["xrd_crystallite_size"] = 0.0

    return features                            # 返回XRD特征


def extract_ir_features_from_file(filepath):
    """从IR CSV文件提取特征（和step1中的逻辑完全一致）"""
    if not os.path.exists(filepath):           # 检查文件是否存在
        print(f"  警告: IR文件不存在: {filepath}")  # 打印警告
        return None                             # 返回None表示缺失

    df = pd.read_csv(filepath, header=0)        # 读取CSV
    cols = df.columns.tolist()                  # 获取列名
    wavenumbers = df[cols[0]].values            # 第一列是波数
    absorbance = np.abs(df[cols[1]].values)     # 第二列是吸光度（取绝对值）

    features = {}                               # 空字典

    # 基本统计量
    features["ir_mean_absorbance"] = np.mean(absorbance)  # 平均吸光度
    features["ir_max_absorbance"] = np.max(absorbance)  # 最大吸光度
    features["ir_std_absorbance"] = np.std(absorbance)  # 标准差
    features["ir_total_area"] = np.trapz(absorbance, wavenumbers)  # 总面积

    # 找峰
    peak_indices, peak_props = find_peaks(     # 找IR谱图中的峰
        absorbance,                            # 吸光度数据
        prominence=0.05,                       # 最低显著性
        distance=10                            # 最小距离
    )
    features["ir_peak_count"] = len(peak_indices)  # 峰的数量

    if len(peak_indices) > 0:
        peak_heights = peak_props["prominences"]
        sorted_idx = np.argsort(peak_heights)[::-1]
        top_n = min(5, len(peak_indices))      # IR最多取5个峰

        for i in range(top_n):
            idx = sorted_idx[i]
            features[f"ir_peak{i+1}_position"] = wavenumbers[peak_indices[idx]]
            features[f"ir_peak{i+1}_height"] = peak_heights[idx]

        for i in range(top_n, 5):              # 补齐到5个
            features[f"ir_peak{i+1}_position"] = 0.0
            features[f"ir_peak{i+1}_height"] = 0.0
    else:
        for i in range(1, 6):
            features[f"ir_peak{i}_position"] = 0.0
            features[f"ir_peak{i}_height"] = 0.0

    # 官能团区域特征
    for group_name, (wn_min, wn_max) in config.IR_FUNCTIONAL_GROUPS.items():  # 遍历官能团
        mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)  # 找范围内的数据
        if np.any(mask):
            features[f"ir_{group_name}_max"] = np.max(absorbance[mask])  # 区域最大吸光度
            features[f"ir_{group_name}_area"] = np.trapz(absorbance[mask], wavenumbers[mask])  # 区域面积
            features[f"ir_{group_name}_mean"] = np.mean(absorbance[mask])  # 区域平均吸光度
        else:
            features[f"ir_{group_name}_max"] = 0.0
            features[f"ir_{group_name}_area"] = 0.0
            features[f"ir_{group_name}_mean"] = 0.0

    # 关键比值
    if "ir_PO4磷酸根_area" in features and "ir_CO3碳酸根_area" in features:
        if features["ir_CO3碳酸根_area"] > 0:  # 避免除以0
            features["ir_ratio_PO4_CO3"] = features["ir_PO4磷酸根_area"] / features["ir_CO3碳酸根_area"]
        else:
            features["ir_ratio_PO4_CO3"] = 0.0
    else:
        features["ir_ratio_PO4_CO3"] = 0.0

    if "ir_酰胺I_area" in features and "ir_酰胺II_area" in features:
        if features["ir_酰胺II_area"] > 0:
            features["ir_ratio_amide1_amide2"] = features["ir_酰胺I_area"] / features["ir_酰胺II_area"]
        else:
            features["ir_ratio_amide1_amide2"] = 0.0
    else:
        features["ir_ratio_amide1_amide2"] = 0.0

    return features                            # 返回IR特征


def extract_sem_single(filepath):
    """从单张SEM图片提取特征（和step1中的extract_sem_single_image完全一致）"""
    if not os.path.exists(filepath):           # 检查文件是否存在
        return None

    image = cv2.imread(filepath)               # 读取图片
    if image is None:
        return None

    image = cv2.resize(image, config.SEM_IMAGE_SIZE, interpolation=cv2.INTER_AREA)  # 缩放到512x512
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度图

    features = {}                              # 空字典

    # 灰度统计
    features["gray_mean"] = np.mean(gray)
    features["gray_std"] = np.std(gray)
    features["gray_median"] = np.median(gray)
    features["gray_min"] = np.min(gray)
    features["gray_max"] = np.max(gray)
    features["gray_skewness"] = stats.skew(gray.flatten())
    features["gray_kurtosis"] = stats.kurtosis(gray.flatten())

    # GLCM纹理
    glcm = graycomatrix(                       # 计算灰度共生矩阵
        gray,
        distances=[config.GLCM_DISTANCE],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=config.GLCM_LEVELS,
        symmetric=True,
        normed=True
    )

    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        prop_values = graycoprops(glcm, prop_name)  # 计算纹理属性
        features[f"glcm_{prop_name}"] = np.mean(prop_values)

    # LBP纹理
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, "uniform")

    n_bins = int(n_points + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    for i in range(n_bins):
        features[f"lbp_{i}"] = hist[i]

    # 形态学特征
    threshold = otsu_threshold(gray)           # Otsu自动阈值
    binary = gray > threshold
    labeled, num_features = nd_label(binary.astype(int))

    features["particle_count"] = num_features

    if num_features > 0:
        areas = []
        for i in range(1, num_features + 1):
            area = np.sum(labeled == i)
            areas.append(area)

        features["morph_mean_area"] = np.mean(areas)
        features["morph_std_area"] = np.std(areas)
        features["morph_max_area"] = np.max(areas)
        features["morph_area_fraction"] = np.sum(binary) / gray.size
        if np.mean(areas) > 0:
            features["morph_area_cv"] = np.std(areas) / np.mean(areas)
        else:
            features["morph_area_cv"] = 0.0
    else:
        features["morph_mean_area"] = 0.0
        features["morph_std_area"] = 0.0
        features["morph_max_area"] = 0.0
        features["morph_area_fraction"] = 0.0
        features["morph_area_cv"] = 0.0

    return features


def extract_sem_from_files(filepaths, feature_prefix):
    """
    从多张SEM图片提取特征并取平均
    filepaths: 文件路径列表
    feature_prefix: "sem_s_"（表面）或 "sem_c_"（截面）
    返回: 带前缀的特征字典，如果所有图片都失败返回None
    """
    all_features = []                          # 存放每张图片的特征

    for fp in filepaths:                       # 遍历每个文件路径
        feat = extract_sem_single(fp)          # 从这张图片提取特征
        if feat is not None:                   # 如果提取成功
            all_features.append(feat)          # 加入列表

    if len(all_features) == 0:                 # 如果所有图片都失败了
        return None

    # 取平均
    feature_keys = list(all_features[0].keys())  # 取特征名
    averaged = {}
    for key in feature_keys:                   # 遍历每个特征
        values = [f[key] for f in all_features]  # 收集所有图片的值
        averaged[f"{feature_prefix}{key}"] = np.mean(values)  # 取平均，加前缀

    return averaged


# ======================== 应用特征选择管道 ========================
def apply_feature_selection(features_dict, selector_info, feature_names_all):
    """
    对预测样品的特征应用和训练时完全一样的特征选择
    确保预测时用的特征和训练时一模一样

    参数:
        features_dict: 从新样品提取的全部特征（字典）
        selector_info: 训练时保存的特征选择器信息
        feature_names_all: 全部特征的名称列表（和训练时顺序一致）
    返回:
        选择后的特征数组（numpy array）
    """
    # 把特征字典转成数组（按照训练时的特征顺序）
    X = np.array([features_dict.get(name, 0.0) for name in feature_names_all])  # 按顺序提取特征值，没有的特征填0

    # 第1步：方差过滤（用训练时保存的过滤器）
    var_thresh = selector_info["var_thresh"]   # 加载方差过滤器
    X_var = var_thresh.transform(X.reshape(1, -1))  # 应用过滤（reshape成1行多列的二维数组）

    # 第2步：互信息选择（如果训练时用了的话）
    if selector_info["mi_selector"] is not None:  # 如果训练时用了互信息选择
        mi_selector = selector_info["mi_selector"]  # 加载互信息选择器
        X_mi = mi_selector.transform(X_var)    # 应用选择
    else:
        X_mi = X_var                           # 没有互信息选择，直接用上一步的结果

    # 第3步：RFECV选择（用训练时保存的选择器）
    rfe = selector_info["rfe"]                 # 加载RFECV选择器
    X_final = rfe.transform(X_mi)              # 应用RFECV选择

    return X_final                             # 返回最终选择的特征


# ======================== 主程序 ========================
if __name__ == "__main__":                     # 当直接运行此脚本时执行
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="预测新龟甲样品的老化级别")  # 创建解析器，添加描述文字

    # 添加可选的参数：--xrd、--ir、--sem_surface、--sem_cross
    # nargs="*" 表示可以跟0个或多个值，default=None表示没传就是None
    parser.add_argument("--xrd", type=str, default=None, help="XRD的CSV文件路径（可选）")  # XRD文件路径，可选
    parser.add_argument("--ir", type=str, default=None, help="IR的CSV文件路径（可选）")  # IR文件路径，可选
    parser.add_argument("--sem_surface", type=str, nargs="*", default=None, help="SEM表面图片路径，可多张（可选）")  # 表面SEM，可选，可多个
    parser.add_argument("--sem_cross", type=str, nargs="*", default=None, help="SEM截面图片路径，可多张（可选）")  # 截面SEM，可选，可多个

    args = parser.parse_args()                 # 解析命令行参数

    # 检查是否至少提供了一个数据源
    if args.xrd is None and args.ir is None and args.sem_surface is None and args.sem_cross is None:
        print("错误: 至少要提供一个数据源！")     # 打印错误
        print("用法示例:")                      # 给出用法
        print("  python step3_预测新样品.py --xrd data/xrd/GH001_xrd.csv --ir data/ir/GH001_ir.csv --sem_surface data/sem/GH001_surface_5000x.jpg --sem_cross data/sem/GH001_cross_10000x.jpg")
        print("  python step3_预测新样品.py --ir data/ir/GH001_ir.csv")
        print("  python step3_预测新样品.py --ir data/ir/GH001_ir.csv --sem_surface data/sem/GH001_surface_5000x.jpg")
        exit()                                 # 退出程序

    print("=" * 60)                            # 打印分隔线
    print("步骤3: 预测新样品")                   # 打印当前步骤
    print("=" * 60)                            # 打印分隔线

    # 检查模型文件是否存在
    model_path = os.path.join(config.MODEL_DIR, "svm_model.joblib")  # 模型文件路径
    if not os.path.exists(model_path):         # 如果模型文件不存在
        print(f"\n错误: 找不到训练好的模型文件: {model_path}")  # 打印错误
        print("请先运行 'python step2_训练模型.py' 训练模型。")  # 告诉用户怎么办
        exit()                                 # 退出程序

    # 加载所有训练好的模型和工具
    print("\n加载训练好的模型...")                # 提示正在加载
    model = joblib.load(model_path)            # 加载SVM模型

    # 加载归一化工具
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.joblib")  # 归一化工具路径
    scaler = joblib.load(scaler_path)          # 加载归一化工具

    # 加载特征选择器
    selector_path = os.path.join(config.MODEL_DIR, "selector.joblib")  # 选择器路径
    selector_info = joblib.load(selector_path)  # 加载选择器信息

    # 加载标签编码器
    encoder_path = os.path.join(config.MODEL_DIR, "label_encoder.joblib")  # 编码器路径
    le = joblib.load(encoder_path)             # 加载标签编码器

    # 获取训练时全部特征的名称
    feature_names_all = selector_info["feature_names_all"]  # 全部原始特征名列表
    print(f"  模型加载完成。训练时使用的特征数: {len(feature_names_all)}")  # 打印特征数

    # 从新样品文件中提取特征
    print(f"\n提取新样品特征:")                    # 打印提示

    # 初始化空特征字典
    all_feat = {}                              # 空字典，存放所有特征

    # 提取XRD特征（如果提供了）
    if args.xrd is not None:                   # 如果传了--xrd参数
        print(f"  XRD: {args.xrd}")             # 打印XRD文件路径
        xrd_feat = extract_xrd_features_from_file(args.xrd)  # 提取XRD特征
        if xrd_feat is not None:               # 如果提取成功
            all_feat.update(xrd_feat)          # 加入XRD特征
            print(f"    提取了 {len(xrd_feat)} 个XRD特征")  # 打印特征数
        else:
            print(f"    [警告] XRD文件无法读取，XRD特征将填0")  # 警告

    # 提取IR特征（如果提供了）
    if args.ir is not None:                    # 如果传了--ir参数
        print(f"  IR:  {args.ir}")              # 打印IR文件路径
        ir_feat = extract_ir_features_from_file(args.ir)  # 提取IR特征
        if ir_feat is not None:                # 如果提取成功
            all_feat.update(ir_feat)           # 加入IR特征
            print(f"    提取了 {len(ir_feat)} 个IR特征")  # 打印特征数
        else:
            print(f"    [警告] IR文件无法读取，IR特征将填0")  # 警告

    # 提取SEM表面特征（如果提供了）
    if args.sem_surface is not None and len(args.sem_surface) > 0:  # 如果传了--sem_surface参数
        print(f"  SEM表面: {len(args.sem_surface)}张图片")  # 打印图片数量
        sem_s_feat = extract_sem_from_files(args.sem_surface, "sem_s_")  # 提取表面特征
        if sem_s_feat is not None:             # 如果提取成功
            all_feat.update(sem_s_feat)        # 加入表面特征
            print(f"    提取了 {len(sem_s_feat)} 个表面SEM特征")  # 打印特征数
        else:
            print(f"    [警告] 所有表面SEM图片无法读取，表面特征将填0")  # 警告

    # 提取SEM截面特征（如果提供了）
    if args.sem_cross is not None and len(args.sem_cross) > 0:  # 如果传了--sem_cross参数
        print(f"  SEM截面: {len(args.sem_cross)}张图片")  # 打印图片数量
        sem_c_feat = extract_sem_from_files(args.sem_cross, "sem_c_")  # 提取截面特征
        if sem_c_feat is not None:             # 如果提取成功
            all_feat.update(sem_c_feat)        # 加入截面特征
            print(f"    提取了 {len(sem_c_feat)} 个截面SEM特征")  # 打印特征数
        else:
            print(f"    [警告] 所有截面SEM图片无法读取，截面特征将填0")  # 警告

    print(f"  共提取了 {len(all_feat)} 个原始特征")  # 打印总特征数量

    # 应用特征选择（和训练时完全一样的流程）
    X_selected = apply_feature_selection(all_feat, selector_info, feature_names_all)  # 特征选择

    # 归一化（用训练时保存的归一化工具）
    X_scaled = scaler.transform(X_selected)    # 归一化特征

    # 用模型预测
    prediction = model.predict(X_scaled)       # 预测类别（数字）
    predicted_label = le.inverse_transform(prediction)[0]  # 把数字标签转回文字，如 0 → "轻度老化"

    # 获取各类别的概率（模型有多大把握）
    probabilities = model.predict_proba(X_scaled)[0]  # 获取每个类别的概率值（一维数组）

    # 打印预测结果
    print("\n" + "=" * 60)                    # 打印分隔线
    print("预测结果:")                           # 打印标题
    print("=" * 60)                           # 打印分隔线
    print(f"\n  预测老化级别: {predicted_label}")  # 打印预测结果

    # 找到预测类别对应的概率（置信度）
    class_idx = list(model.classes_).index(prediction[0])  # 找到预测类别在classes_中的索引
    confidence = probabilities[class_idx] * 100  # 转成百分比
    print(f"  置信度: {confidence:.1f}%")       # 打印置信度

    # 打印每个类别的概率
    print(f"\n  各类别概率:")                    # 打印标题
    for i, cls in enumerate(model.classes_):   # 遍历每个类别
        cls_name = le.inverse_transform([cls])[0]  # 把数字标签转成文字
        print(f"    {cls_name}: {probabilities[i]*100:.1f}%")  # 打印每个类别的概率

    # 根据置信度给出判断建议
    if confidence >= 80:                        # 置信度很高
        print(f"\n  [OK] 模型非常有把握，该样品很可能是 '{predicted_label}'")  # 肯定的结论
    elif confidence >= 60:                      # 置信度中等
        print(f"\n  [注意] 模型比较有信心，但建议结合专业知识进一步确认")  # 中等把握
    else:                                       # 置信度低
        print(f"\n  [警告] 模型不太确定，建议结合多种手段综合判断")  # 把握不大

    # 提醒缺失的数据源
    if args.xrd is None:
        print(f"\n  [注意] 本次预测没有XRD数据，结果仅供参考")  # 提醒
    if args.ir is None:
        print(f"  [注意] 本次预测没有IR数据，结果仅供参考")  # 提醒
    if args.sem_surface is None and args.sem_cross is None:
        print(f"  [注意] 本次预测没有SEM数据，结果仅供参考")  # 提醒

    print(f"\n{'=' * 60}")                    # 打印分隔线
    print("预测完成!")                             # 打印完成提示
    print(f"{'=' * 60}")                      # 打印分隔线
