# ============================================================
# step1_提取特征.py —— 从原始数据中提取有用指标
# ============================================================
# 这一步做什么：
#   1. 读取你的 XRD CSV、IR CSV、SEM 图片文件
#   2. 从 XRD 谱图中提取峰位、结晶度等指标
#   3. 从 IR 谱图中提取峰位、官能团强度等指标
#   4. 从 SEM 图片中提取纹理、形态等指标（分表面和截面）
#   5. 把所有指标拼成一张表格，保存为 features.csv
#
# 数据要求（不强制三种都有，有几种就用几种）：
#   1. XRD: data/xrd/样品名_xrd.csv （可选）
#   2. IR:  data/ir/样品名_ir.csv （可选）
#   3. SEM: data/sem/样品名_surface_xxx.jpg （表面，可选，可多张）
#          data/sem/样品名_cross_xxx.jpg   （截面，可选，可多张）
#   4. data/sample_labels.csv 写清楚每个样品的老化级别
#   5. 运行: python step1_提取特征.py
#
# 运行后生成:
#   features.csv —— 每个样品一行，每列是一个特征指标
# ============================================================

# ======================== 导入需要的库 ========================
import os                                     # 操作系统接口，用来找文件、创建文件夹
import re                                     # 正则表达式模块，用来从文件名中提取信息
import numpy as np                            # 数值计算库，用来做数学运算
import pandas as pd                           # 表格处理库，用来读写CSV和处理数据表
from scipy.signal import find_peaks           # 从scipy导入找峰的函数，用于在谱图中找峰值
from scipy.ndimage import label as nd_label   # 从scipy导入标记函数，用于SEM图片中数颗粒数量
from scipy import stats                       # 从scipy导入统计函数，用来算偏度、峰度等
from skimage.feature import graycomatrix, graycoprops  # 从skimage导入GLCM纹理分析工具（灰度共生矩阵）
from skimage.feature import local_binary_pattern         # 从skimage导入LBP（局部二值模式）纹理工具
from skimage.filters import threshold_otsu as otsu_threshold  # Otsu自动阈值分割
import cv2                                    # OpenCV图像处理库，用来读取和处理图片
import config                                 # 导入你的配置文件，里面设置了各种参数


# ======================== 文件名解析 ========================
def extract_sample_prefix(filename):
    """
    从SEM文件名中提取样品名前缀
    比如: "GH001_surface_5000x.jpg" → "GH001"
         "sample_A_cross_10000x.tif" → "sample_A"
    原理: 找第一个包含 "surface"/"cross" 等关键词之前的部分作为前缀
    """
    # 去掉文件扩展名（.jpg、.tif 等）
    name_no_ext = filename
    for ext in config.SEM_IMAGE_EXTENSIONS:  # 遍历所有支持的扩展名
        if name_no_ext.lower().endswith(ext):  # 如果文件名以这个扩展名结尾
            name_no_ext = name_no_ext[:len(name_no_ext) - len(ext)]  # 去掉扩展名
            break                               # 找到就跳出循环

    # 把文件名全部转成小写（不区分大小写地找关键词）
    name_lower = name_no_ext.lower()

    # 尝试找表面/截面关键词的位置
    keywords = config.SEM_SURFACE_KEYWORDS + config.SEM_CROSS_KEYWORDS  # 合并两类关键词
    min_pos = len(name_lower)                   # 初始化为最大可能位置
    for kw in keywords:                         # 遍历每个关键词
        pos = name_lower.find(kw)               # 找关键词在文件名中的位置
        if pos != -1 and pos < min_pos:         # 如果找到了，而且比之前找到的更靠前
            min_pos = pos                       # 记录这个位置

    if min_pos < len(name_lower):               # 如果找到了关键词
        prefix = name_no_ext[:min_pos].rstrip("_")  # 取关键词之前的部分作为前缀，去掉末尾的下划线
        return prefix                           # 返回样品名前缀
    else:
        return None                             # 没找到关键词，返回空


def classify_sem_file(filename):
    """
    判断一个SEM文件是表面还是截面
    返回: "surface"（表面）、"cross"（截面）、"unknown"（无法识别）
    """
    name_lower = filename.lower()               # 文件名转小写

    for kw in config.SEM_SURFACE_KEYWORDS:      # 遍历表面关键词
        if kw.lower() in name_lower:            # 如果文件名包含这个关键词
            return "surface"                     # 判定为表面图

    for kw in config.SEM_CROSS_KEYWORDS:        # 遍历截面关键词
        if kw.lower() in name_lower:            # 如果文件名包含这个关键词
            return "cross"                       # 判定为截面图

    return "unknown"                            # 两个都不包含，无法识别


# ======================== 发现样品 ========================
def discover_samples():
    """
    从标签文件读取所有样品，然后检查每个样品有哪些数据
    不要求三种数据都有，有几种就算几种

    返回: 一个字典，结构如下：
    {
        "GH001": {
            "has_xrd": True/False,           # 有没有XRD数据
            "has_ir": True/False,            # 有没有IR数据
            "sem_surface_files": ["文件名1", "文件名2", ...],  # 表面SEM图片列表
            "sem_cross_files": ["文件名1", ...],               # 截面SEM图片列表
        },
        ...
    }
    """
    # 先检查标签文件是否存在
    if not os.path.exists(config.LABEL_FILE):  # 如果标签文件不存在
        print(f"错误: 找不到标签文件 {config.LABEL_FILE}")  # 打印错误信息
        print("请在 data/ 目录下创建 sample_labels.csv")    # 告诉用户怎么办
        print("格式: sample_name,label")       # 给出格式示例
        print("例如: GH001,轻度老化")          # 给出具体例子
        return {}                               # 返回空字典

    # 读取标签文件，获取所有样品名
    labels_df = pd.read_csv(config.LABEL_FILE)  # 读取标签CSV
    sample_names = labels_df["sample_name"].tolist()  # 提取样品名列，转成列表

    # 列出所有文件夹中的文件
    xrd_files = os.listdir(config.XRD_DIR) if os.path.exists(config.XRD_DIR) else []  # 列出xrd文件夹里的文件（如果文件夹不存在就是空列表）
    ir_files = os.listdir(config.IR_DIR) if os.path.exists(config.IR_DIR) else []     # 列出ir文件夹里的文件
    sem_files = os.listdir(config.SEM_DIR) if os.path.exists(config.SEM_DIR) else []  # 列出sem文件夹里的文件

    # 建立XRD和IR的文件名集合（用于快速查找）
    # XRD: "GH001_xrd.csv" → 前缀是 "GH001"
    xrd_prefixes = set()                        # 空集合，存放XRD样品前缀
    for f in xrd_files:                         # 遍历每个xrd文件
        if f.lower().endswith("_xrd.csv"):      # 只处理以 _xrd.csv 结尾的文件（不区分大小写）
            prefix = f.replace("_xrd.csv", "").replace("_XRD.csv", "")  # 去掉后缀得到前缀
            xrd_prefixes.add(prefix)           # 加入集合

    # IR: "GH001_ir.csv" → 前缀是 "GH001"
    ir_prefixes = set()                         # 空集合，存放IR样品前缀
    for f in ir_files:                          # 遍历每个ir文件
        if f.lower().endswith("_ir.csv"):       # 只处理以 _ir.csv 结尾的文件
            low = f.lower()                     # 转小写后截取
            prefix = f[:len(f) - len("_ir.csv")]  # 去掉后缀得到前缀
            ir_prefixes.add(prefix)            # 加入集合

    # 处理SEM文件：按样品名和类型（表面/截面）分组
    sem_by_sample = {}                          # 字典: 样品名 → {"surface": [文件名], "cross": [文件名]}
    for f in sem_files:                         # 遍历每个sem文件
        # 检查是不是支持的图片格式
        is_image = False                        # 标记是否为图片文件
        for ext in config.SEM_IMAGE_EXTENSIONS:  # 遍历支持的扩展名
            if f.lower().endswith(ext):         # 如果文件名以这个扩展名结尾
                is_image = True                 # 标记为图片
                break                           # 跳出扩展名循环

        if not is_image:                        # 如果不是图片文件，跳过
            continue

        # 判断是表面还是截面
        sem_type = classify_sem_file(f)         # 调用分类函数
        if sem_type == "unknown":               # 如果无法识别类型
            print(f"  警告: 无法识别SEM文件类型: {f}，跳过")  # 打印警告
            continue

        # 提取样品名前缀
        prefix = extract_sample_prefix(f)       # 从文件名提取前缀
        if prefix is None:                      # 如果提取失败
            print(f"  警告: 无法提取样品名前缀: {f}，跳过")  # 打印警告
            continue

        # 把这个文件加入对应样品的列表中
        if prefix not in sem_by_sample:         # 如果这个样品名还没出现过
            sem_by_sample[prefix] = {"surface": [], "cross": []}  # 创建空列表

        sem_by_sample[prefix][sem_type].append(f)  # 把文件名加入对应的类型列表

    # 为每个标签中的样品检查有哪些数据
    sample_info = {}                            # 最终的结果字典
    for sample in sample_names:                 # 遍历每个样品
        sample_info[sample] = {
            "has_xrd": sample in xrd_prefixes,  # True表示有XRD数据
            "has_ir": sample in ir_prefixes,    # True表示有IR数据
            "sem_surface_files": sem_by_sample.get(sample, {}).get("surface", []),  # 表面SEM文件列表（没有就是空列表）
            "sem_cross_files": sem_by_sample.get(sample, {}).get("cross", []),      # 截面SEM文件列表
        }

    return sample_info                          # 返回完整的样品信息字典


# ======================== 打印数据可用性统计 ========================
def print_data_availability(sample_info):
    """
    打印每个数据源有多少样品有数据，以及各类数据组合的样品数量
    帮你了解数据完整性
    """
    total = len(sample_info)                    # 总样品数
    if total == 0:                              # 如果没有样品
        return

    # 统计每个数据源有多少样品
    xrd_count = sum(1 for s in sample_info.values() if s["has_xrd"])  # 有XRD的样品数
    ir_count = sum(1 for s in sample_info.values() if s["has_ir"])    # 有IR的样品数
    sem_s_count = sum(1 for s in sample_info.values() if len(s["sem_surface_files"]) > 0)  # 有表面SEM的样品数
    sem_c_count = sum(1 for s in sample_info.values() if len(s["sem_cross_files"]) > 0)    # 有截面SEM的样品数
    sem_s_files = sum(len(s["sem_surface_files"]) for s in sample_info.values())  # 表面SEM图片总数
    sem_c_files = sum(len(s["sem_cross_files"]) for s in sample_info.values())    # 截面SEM图片总数

    print(f"\n数据可用性统计 (共{total}个样品):")        # 打印标题
    print(f"  XRD:       {xrd_count}个样品有数据")        # 打印XRD统计
    print(f"  IR:        {ir_count}个样品有数据")         # 打印IR统计
    print(f"  SEM表面:   {sem_s_count}个样品有数据 (共{sem_s_files}张图)")  # 打印表面SEM统计
    print(f"  SEM截面:   {sem_c_count}个样品有数据 (共{sem_c_files}张图)")  # 打印截面SEM统计

    # 统计各种数据组合
    complete = sum(1 for s in sample_info.values() if s["has_xrd"] and s["has_ir"] and (len(s["sem_surface_files"]) > 0 or len(s["sem_cross_files"]) > 0))  # 三种都有的样品数
    no_sem = sum(1 for s in sample_info.values() if s["has_xrd"] and s["has_ir"] and len(s["sem_surface_files"]) == 0 and len(s["sem_cross_files"]) == 0)  # 只有XRD+IR的样品数
    no_xrd = sum(1 for s in sample_info.values() if not s["has_xrd"] and s["has_ir"] and (len(s["sem_surface_files"]) > 0 or len(s["sem_cross_files"]) > 0))  # 只有IR+SEM的样品数
    ir_only = sum(1 for s in sample_info.values() if not s["has_xrd"] and s["has_ir"] and len(s["sem_surface_files"]) == 0 and len(s["sem_cross_files"]) == 0)  # 只有IR的样品数

    print(f"\n  数据组合:")                         # 打印标题
    print(f"    XRD+IR+SEM: {complete}个样品")       # 打印三种都有的数量
    print(f"    XRD+IR:     {no_sem}个样品")         # 打印只有XRD+IR的数量
    print(f"    IR+SEM:     {no_xrd}个样品")         # 打印只有IR+SEM的数量
    print(f"    只有IR:     {ir_only}个样品")         # 打印只有IR的数量

    # 提醒缺失数据的样品
    missing_any = [name for name, info in sample_info.items()
                   if not info["has_xrd"] or not info["has_ir"] or
                   (len(info["sem_surface_files"]) == 0 and len(info["sem_cross_files"]) == 0)]
    if len(missing_any) > 0:                    # 如果有样品缺少某种数据
        print(f"\n  [注意] 以下{len(missing_any)}个样品数据不全: {', '.join(missing_any)}")  # 列出数据不全的样品
        print(f"    缺少的那部分特征会自动填0，不影响模型训练")  # 说明处理方式


# ======================== XRD 特征提取 ========================
def extract_xrd_features(sample_name):
    """
    从一个 XRD CSV 文件中提取特征指标
    XRD谱图告诉我们龟甲内部的晶体结构——老化越深，结晶度越变化

    提取的特征包括：
    - 峰的数量、位置、高度（谱图上凸起的位置和大小）
    - 结晶度指数（晶体部分占总体的比例，老化会降低结晶度）
    - 最强峰的信息（最突出的那个峰的位置和宽度）
    - 谱图整体统计（均值、最大值、波动程度等）

    参数:
        sample_name: 样品名前缀，如 'GH001'
    返回:
        字典，键是特征名，值是特征数值
    """
    # 拼接完整的文件路径
    filepath = os.path.join(config.XRD_DIR, sample_name + "_xrd.csv")  # 如: data/xrd/GH001_xrd.csv

    # 读取CSV文件，假设第一列是角度(2θ)，第二列是强度
    # header=0 表示第一行是表头（列名）
    df = pd.read_csv(filepath, header=0)        # 读取CSV文件成数据表

    # 自动识别哪列是角度、哪列是强度
    # 因为不同人保存的CSV列名可能不一样，这里灵活处理
    cols = df.columns.tolist()                  # 获取所有列名，转成列表
    angle_col = cols[0]                         # 假设第一列是角度（2θ）
    intensity_col = cols[1]                     # 假设第二列是强度

    # 把两列数据提取成numpy数组（方便做数学运算）
    angles = df[angle_col].values               # 角度数据，如 [10, 10.1, 10.2, ...]
    intensities = df[intensity_col].values      # 强度数据，如 [1200, 1250, 1180, ...]

    # 初始化一个空字典，用来存放所有提取出来的特征
    features = {}                               # 空字典，后面会往里添加键值对

    # --- 特征1: 谱图基本统计量 ---
    features["xrd_mean_intensity"] = np.mean(intensities)  # 平均强度：所有数据点强度的平均值
    features["xrd_max_intensity"] = np.max(intensities)    # 最大强度：最强的那个点的强度值
    features["xrd_std_intensity"] = np.std(intensities)    # 强度标准差：数据波动程度，越大说明峰越尖锐
    features["xrd_total_area"] = np.trapz(intensities, angles)  # 谱图总面积：曲线下的面积，用梯形法积分

    # 偏度和峰度：描述数据分布的形状
    # 偏度>0说明右边拖尾（有高角度尾巴），偏度<0说明左边拖尾
    if len(intensities) > 2:                    # 至少有3个数据点才能算偏度
        features["xrd_skewness"] = stats.skew(intensities)  # 偏度：分布不对称的程度
    else:
        features["xrd_skewness"] = 0.0          # 数据太少，默认0

    # 峰度：描述分布是"尖"还是"平"
    # 峰度>0说明比正态分布更尖（有尖锐的峰），峰度<0说明更平坦
    if len(intensities) > 3:                    # 至少有4个数据点才能算峰度
        features["xrd_kurtosis"] = stats.kurtosis(intensities)  # 峰度：分布尖锐或平坦的程度
    else:
        features["xrd_kurtosis"] = 0.0          # 数据太少，默认0

    # --- 特征2: 找峰 ---
    # find_peaks: 自动在谱图中找峰值
    # prominence: 最低显著性，值越大找到的峰越少、越明显
    # distance: 两个峰之间最少要隔多少个数据点
    peak_indices, peak_props = find_peaks(      # 调用找峰函数，返回峰的索引和属性
        intensities,                            # 输入强度数据
        prominence=config.XRD_PEAK_PROMINENCE,  # 最低显著性，从config读取
        distance=config.XRD_PEAK_DISTANCE       # 峰之间最小距离，从config读取
    )

    features["xrd_peak_count"] = len(peak_indices)  # 峰的数量：找到了几个峰

    # 如果找到了峰，就提取每个峰的详细信息
    if len(peak_indices) > 0:                   # 如果有找到峰的话
        # 获取每个峰的高度（从 find_peaks 返回的属性中读取）
        peak_heights = peak_props["prominences"]  # 每个峰的显著性（峰有多"突出"）

        # 取最强的3个峰的信息（按显著性排序）
        # argsort返回从小到大的索引，[::-1]反转变成从大到小
        sorted_idx = np.argsort(peak_heights)[::-1]  # 按显著性从高到低排序的索引
        top_n = min(3, len(peak_indices))       # 最多取3个，如果峰不够3个就取实际数量

        for i in range(top_n):                  # 遍历最强的几个峰
            idx = sorted_idx[i]                 # 第i强的峰在peak_indices中的位置
            peak_pos = angles[peak_indices[idx]]  # 这个峰对应的角度位置（2θ值）
            peak_h = peak_heights[idx]          # 这个峰的显著性
            features[f"xrd_peak{i+1}_position"] = peak_pos  # 保存第i强峰的角度位置
            features[f"xrd_peak{i+1}_height"] = peak_h  # 保存第i强峰的显著性

        # 如果不足3个峰，剩下的位置填0（保证所有样品的特征列数一致）
        for i in range(top_n, 3):               # 如果只有1或2个峰，补齐到3个
            features[f"xrd_peak{i+1}_position"] = 0.0  # 位置填0
            features[f"xrd_peak{i+1}_height"] = 0.0  # 高度填0
    else:
        # 如果完全没找到峰，所有峰相关特征都填0
        features["xrd_peak1_position"] = 0.0    # 最强峰位置，没有峰就填0
        features["xrd_peak1_height"] = 0.0      # 最强峰高度
        features["xrd_peak2_position"] = 0.0    # 第二强峰
        features["xrd_peak2_height"] = 0.0
        features["xrd_peak3_position"] = 0.0    # 第三强峰
        features["xrd_peak3_height"] = 0.0

    # --- 特征3: 结晶度指数 ---
    # 结晶度 = 结晶区域的面积 / (结晶区域面积 + 非晶区域面积)
    # 老化过程中，晶体结构会被破坏，结晶度下降
    crystalline_area = 0                        # 结晶区域总面积，初始为0
    for region in config.XRD_CRYSTALLINE_REGIONS:  # 遍历每个结晶区域
        # 找到在这个角度范围内的数据点
        mask = (angles >= region[0]) & (angles <= region[1])  # 布尔掩码：True表示在这个范围内的点
        if np.any(mask):                        # 如果范围内有数据点
            crystalline_area += np.trapz(intensities[mask], angles[mask])  # 积分计算该区域的面积

    amorphous_mask = (angles >= config.XRD_AMORPHOUS_REGION[0]) & (angles <= config.XRD_AMORPHOUS_REGION[1])
    if np.any(amorphous_mask):                  # 如果非晶区域有数据点
        amorphous_area = np.trapz(intensities[amorphous_mask], angles[amorphous_mask])  # 积分计算非晶区域面积
    else:
        amorphous_area = 0.0                    # 没有数据点就设为0

    # 结晶度 = 结晶面积 / (结晶面积 + 非晶面积)，结果在0~1之间
    total = crystalline_area + amorphous_area   # 总区域面积
    if total > 0:                               # 如果总面积大于0（避免除以0的错误）
        features["xrd_crystallinity_index"] = crystalline_area / total  # 结晶度指数
    else:
        features["xrd_crystallinity_index"] = 0.0  # 没有数据，结晶度设为0

    # --- 特征4: 最强峰的半高宽(FWHM) ---
    # FWHM = Full Width at Half Maximum，峰的"胖瘦"程度
    # 老化会让峰变宽（晶体变小、缺陷增多）
    if len(peak_indices) > 0:                   # 如果有找到峰
        # 找到最强峰的位置
        main_peak_idx = peak_indices[sorted_idx[0]]  # 最强峰在angles数组中的索引
        main_peak_pos = angles[main_peak_idx]   # 最强峰的角度位置
        main_peak_height = intensities[main_peak_idx]  # 最强峰的高度

        # 半高 = 峰顶高度的一半
        half_max = main_peak_height / 2.0       # 计算半高值

        # 从峰顶向左找，找到第一个低于半高的位置
        left_idx = main_peak_idx                # 从峰顶开始向左搜索
        while left_idx > 0 and intensities[left_idx] > half_max:  # 向左遍历，直到强度低于半高
            left_idx -= 1                       # 索引减1，继续向左

        # 从峰顶向右找，找到第一个低于半高的位置
        right_idx = main_peak_idx               # 从峰顶开始向右搜索
        while right_idx < len(intensities) - 1 and intensities[right_idx] > half_max:  # 向右遍历
            right_idx += 1                      # 索引加1，继续向右

        # 半高宽 = 右边位置 - 左边位置（角度差）
        features["xrd_fwhm"] = angles[right_idx] - angles[left_idx]  # 半高宽，单位是角度(度)

        # Scherrer公式估算晶粒尺寸（可选，需要知道X射线波长）
        # D = K * λ / (β * cosθ)，其中β是FWHM（弧度制）
        fwhm_rad = np.deg2rad(features["xrd_fwhm"])  # 把角度制的FWHM转成弧度制
        if fwhm_rad > 0:                        # 避免除以0
            features["xrd_crystallite_size"] = 0.9 * 1.5406 / (fwhm_rad * np.cos(np.deg2rad(main_peak_pos)))  # Scherrer公式算晶粒尺寸(nm)
        else:
            features["xrd_crystallite_size"] = 0.0  # FWHM为0，晶粒尺寸设为0
    else:
        features["xrd_fwhm"] = 0.0              # 没有峰，FWHM为0
        features["xrd_crystallite_size"] = 0.0  # 没有峰，晶粒尺寸为0

    return features                             # 返回这个样品的所有XRD特征


# ======================== IR 特征提取 ========================
def extract_ir_features(sample_name):
    """
    从一个 IR CSV 文件中提取特征指标
    IR光谱告诉我们龟甲中化学成分的变化——老化后蛋白质降解、矿物成分改变

    提取的特征包括：
    - 峰的数量、位置
    - 各个官能团区域的峰值（磷酸根、碳酸根、酰胺等）
    - 关键比值（PO4/CO3、酰胺I/II）——老化会导致这些比值变化
    - 谱图整体统计
    """
    filepath = os.path.join(config.IR_DIR, sample_name + "_ir.csv")  # 拼接完整文件路径

    df = pd.read_csv(filepath, header=0)        # 读取CSV文件
    cols = df.columns.tolist()                  # 获取列名列表
    wavenumber_col = cols[0]                    # 第一列是波数(cm⁻¹)
    absorbance_col = cols[1]                    # 第二列是吸光度

    wavenumbers = df[wavenumber_col].values     # 波数数据，如 [4000, 3999, 3998, ...]
    absorbance = df[absorbance_col].values      # 吸光度数据

    # 简单预处理：把吸光度取绝对值（有些仪器输出的是负值）
    absorbance = np.abs(absorbance)             # 取绝对值，确保数据是正的

    features = {}                               # 空字典，存放IR特征

    # --- 特征1: 谱图基本统计量 ---
    features["ir_mean_absorbance"] = np.mean(absorbance)  # 平均吸光度
    features["ir_max_absorbance"] = np.max(absorbance)    # 最大吸光度
    features["ir_std_absorbance"] = np.std(absorbance)    # 吸光度标准差（波动程度）
    features["ir_total_area"] = np.trapz(absorbance, wavenumbers)  # 谱图总面积

    # --- 特征2: 找峰 ---
    # IR谱图的峰是吸收峰（吸光度大的地方），所以和XRD一样用find_peaks找"凸起"
    peak_indices, peak_props = find_peaks(      # 找IR谱图中的吸收峰
        absorbance,                             # 输入吸光度数据
        prominence=0.05,                        # 最低显著性0.05（IR峰的显著性通常比XRD小）
        distance=10                             # 峰之间最少隔10个数据点
    )

    features["ir_peak_count"] = len(peak_indices)  # 峰的数量

    if len(peak_indices) > 0:                   # 如果找到了峰
        peak_heights = peak_props["prominences"]  # 每个峰的显著性
        sorted_idx = np.argsort(peak_heights)[::-1]  # 按显著性从高到低排序
        top_n = min(5, len(peak_indices))       # 最多取5个最强峰（IR的峰比XRD多）

        for i in range(top_n):                  # 遍历最强的几个峰
            idx = sorted_idx[i]                 # 第i强的峰的索引
            peak_pos = wavenumbers[peak_indices[idx]]  # 这个峰的波数位置
            peak_h = peak_heights[idx]          # 这个峰的显著性
            features[f"ir_peak{i+1}_position"] = peak_pos  # 保存第i强峰的波数
            features[f"ir_peak{i+1}_height"] = peak_h  # 保存第i强峰的显著性

        # 不足的补齐到5个
        for i in range(top_n, 5):               # 如果峰不够5个，补0
            features[f"ir_peak{i+1}_position"] = 0.0
            features[f"ir_peak{i+1}_height"] = 0.0
    else:
        # 没有峰就全部填0
        for i in range(1, 6):                   # 循环1到5
            features[f"ir_peak{i}_position"] = 0.0
            features[f"ir_peak{i}_height"] = 0.0

    # --- 特征3: 官能团区域特征 ---
    # 对每个官能团区域，提取该区域内的最大吸光度和面积
    # 这些区域对应龟甲中的不同化学成分
    for group_name, (wn_min, wn_max) in config.IR_FUNCTIONAL_GROUPS.items():  # 遍历每个官能团
        # 找到在这个波数范围内的数据点
        mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)  # 布尔掩码：True表示在这个范围内的点
        if np.any(mask):                        # 如果范围内有数据点
            # 该区域的最大吸光度（这个官能团最强吸收的强度）
            features[f"ir_{group_name}_max"] = np.max(absorbance[mask])
            # 该区域的面积（这个官能团的总量）
            features[f"ir_{group_name}_area"] = np.trapz(absorbance[mask], wavenumbers[mask])
            # 该区域的平均吸光度
            features[f"ir_{group_name}_mean"] = np.mean(absorbance[mask])
        else:
            # 范围内没有数据，全部填0
            features[f"ir_{group_name}_max"] = 0.0
            features[f"ir_{group_name}_area"] = 0.0
            features[f"ir_{group_name}_mean"] = 0.0

    # --- 特征4: 关键比值 ---
    # PO4/CO3比值：反映矿物成分中磷酸盐和碳酸盐的比例，老化过程中会变化
    # 使用 area 来计算比值（面积比峰值更稳定）
    if "ir_PO4磷酸根_area" in features and "ir_CO3碳酸根_area" in features:  # 如果两个区域都有数据
        po4_area = features["ir_PO4磷酸根_area"]  # PO4区域面积
        co3_area = features["ir_CO3碳酸根_area"]  # CO3区域面积
        if co3_area > 0:                        # 避免除以0
            features["ir_ratio_PO4_CO3"] = po4_area / co3_area  # PO4/CO3比值
        else:
            features["ir_ratio_PO4_CO3"] = 0.0  # CO3面积为0，比值设为0
    else:
        features["ir_ratio_PO4_CO3"] = 0.0      # 区域不存在，比值设为0

    # 酰胺I/酰胺II比值：反映蛋白质的结构变化，老化导致胶原蛋白降解
    if "ir_酰胺I_area" in features and "ir_酰胺II_area" in features:  # 如果两个区域都有数据
        amide1 = features["ir_酰胺I_area"]      # 酰胺I区域面积
        amide2 = features["ir_酰胺II_area"]     # 酰胺II区域面积
        if amide2 > 0:                          # 避免除以0
            features["ir_ratio_amide1_amide2"] = amide1 / amide2  # 酰胺I/II比值
        else:
            features["ir_ratio_amide1_amide2"] = 0.0  # 酰胺II为0，比值设为0
    else:
        features["ir_ratio_amide1_amide2"] = 0.0  # 区域不存在，比值设为0

    return features                             # 返回这个样品的所有IR特征


# ======================== SEM 特征提取 ========================
def extract_sem_single_image(filepath):
    """
    从一张 SEM 图片中提取特征指标
    SEM照片展示龟甲表面/截面微观形貌——老化后变得更粗糙、更破碎

    提取的特征包括：
    - GLCM纹理特征：对比度、均匀性、能量等（描述图片纹理的粗糙/平滑程度）
    - LBP纹理特征：局部二值模式直方图（描述微小纹理的分布）
    - 形态学特征：颗粒数量、平均面积等（描述表面有多少"碎片"）
    - 灰度统计：均值、标准差等（描述整体明暗和对比度）

    参数:
        filepath: SEM图片的完整文件路径
    返回:
        字典，键是特征名（不带前缀），值是特征数值
    """
    # 读取图片（OpenCV默认读取为BGR格式，形状为 高×宽×3通道）
    image = cv2.imread(filepath)                # 读取JPG/TIF/PNG图片

    if image is None:                           # 如果图片读取失败（文件不存在或损坏）
        return None                             # 返回None表示失败

    # 把图片统一缩放到固定大小，保证所有图片的特征可以比较
    # INTER_AREA: 适合缩小的插值方法
    image = cv2.resize(image, config.SEM_IMAGE_SIZE, interpolation=cv2.INTER_AREA)  # 缩放到512x512

    # 转成灰度图（SEM图片本来是黑白的，但JPG可能存成彩色格式）
    # cvtColor: 颜色空间转换，BGR转GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转成灰度图，每个像素一个灰度值(0-255)

    features = {}                               # 空字典，存放SEM特征

    # --- 特征1: 灰度统计 ---
    features["gray_mean"] = np.mean(gray)       # 平均灰度：整体明暗程度
    features["gray_std"] = np.std(gray)         # 灰度标准差：对比度大小
    features["gray_median"] = np.median(gray)   # 灰度中位数
    features["gray_min"] = np.min(gray)         # 最小灰度（最暗的点）
    features["gray_max"] = np.max(gray)         # 最大灰度（最亮的点）

    # 偏度和峰度：灰度分布的形状
    features["gray_skewness"] = stats.skew(gray.flatten())  # 偏度：灰度分布是否偏向亮或暗
    features["gray_kurtosis"] = stats.kurtosis(gray.flatten())  # 峰度：灰度分布是尖锐还是平坦

    # --- 特征2: GLCM纹理特征 ---
    # GLCM（灰度共生矩阵）：统计"相邻两个像素的灰度组合"出现的频率
    # 反映纹理的粗糙程度——老化后表面更粗糙，对比度会增大
    glcm = graycomatrix(                        # 计算灰度共生矩阵
        gray,                                   # 输入灰度图
        distances=[config.GLCM_DISTANCE],       # 计算距离（1个像素间隔）
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],  # 计算4个方向（水平、45°、垂直、135°）
        levels=config.GLCM_LEVELS,              # 灰度级数（256级）
        symmetric=True,                         # 对称矩阵（A-B和B-A算同一种）
        normed=True                             # 归一化（数值在0-1之间）
    )

    # 从GLCM中提取5个常用纹理指标
    # 每个指标是一个四维数组 [距离, 角度, _, _]，我们取平均值作为最终特征
    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        # graycoprops: 从GLCM计算指定的纹理属性
        prop_values = graycoprops(glcm, prop_name)  # 计算该属性，返回一个数组
        features[f"glcm_{prop_name}"] = np.mean(prop_values)  # 对所有方向和距离取平均

    # --- 特征3: LBP（局部二值模式）纹理特征 ---
    # LBP：比较每个像素和周围像素的大小关系，编码成二进制模式
    # 反映微观纹理的细节分布
    radius = 1                                  # LBP的半径（1个像素）
    n_points = 8 * radius                       # 周围比较的点数（8个）
    method = "uniform"                          # "uniform"方法：只保留"均匀"模式，减少特征维度

    lbp = local_binary_pattern(gray, n_points, radius, method)  # 计算LBP图

    # LBP的"uniform"模式产生的值范围是 0 到 n_points+1（即0~9）
    # 统计每种模式出现的频率，作为特征
    n_bins = int(n_points + 2)                  # 直方图的bins数量（10个）
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))  # 统计LBP值的分布直方图
    hist = hist.astype("float")                 # 转成浮点数
    hist /= (hist.sum() + 1e-7)                 # 归一化（总和为1，+1e-7避免除以0）

    for i in range(n_bins):                     # 遍历每个bin
        features[f"lbp_{i}"] = hist[i]          # 保存第i个bin的频率

    # --- 特征4: 形态学特征 ---
    # 用Otsu阈值把图片分成"前景"和"背景"两部分
    # otsu_threshold: 自动找最佳分割阈值
    threshold = otsu_threshold(gray)            # Otsu方法自动找阈值
    binary = gray > threshold                   # 大于阈值的为True（前景），小于的为False（背景）

    # 统计连通区域（独立的"颗粒"或"碎片"）
    # nd_label: 给每个连通的区域标上不同的数字
    labeled, num_features = nd_label(binary.astype(int))  # 标记连通区域，返回标记图和区域数量

    features["particle_count"] = num_features   # 颗粒/碎片数量

    if num_features > 0:                        # 如果检测到了颗粒
        # 计算每个颗粒的面积（占的像素数）
        areas = []                              # 空列表，存放每个颗粒的面积
        for i in range(1, num_features + 1):    # 遍历每个颗粒（标记从1开始）
            area = np.sum(labeled == i)         # 统计标记为i的像素数 = 面积
            areas.append(area)                  # 加入列表

        features["morph_mean_area"] = np.mean(areas)  # 平均颗粒面积
        features["morph_std_area"] = np.std(areas)  # 面积标准差
        features["morph_max_area"] = np.max(areas)  # 最大颗粒面积
        features["morph_area_fraction"] = np.sum(binary) / gray.size  # 前景面积占比

        # 颗粒面积越小、数量越多，说明表面越破碎——老化程度越深
        # 变异系数 = 标准差/均值，描述颗粒大小的不均匀程度
        if np.mean(areas) > 0:                  # 避免除以0
            features["morph_area_cv"] = np.std(areas) / np.mean(areas)  # 面积变异系数
        else:
            features["morph_area_cv"] = 0.0
    else:
        # 没有检测到颗粒，全部填0
        features["morph_mean_area"] = 0.0
        features["morph_std_area"] = 0.0
        features["morph_max_area"] = 0.0
        features["morph_area_fraction"] = 0.0
        features["morph_area_cv"] = 0.0

    return features                             # 返回这张图片的所有SEM特征


def extract_sem_for_sample(sem_files, feature_prefix):
    """
    从一个样品的多张同类型SEM图片中提取特征，然后取平均
    比如: 表面的3张图 → 每张提取特征 → 3张的平均值 = 该样品表面特征
          截面的2张图 → 每张提取特征 → 2张的平均值 = 该样品截面特征

    参数:
        sem_files: 文件名列表，如 ["GH001_surface_5000x.jpg", "GH001_surface_10000x.jpg"]
        feature_prefix: 特征前缀，"sem_s_"（表面）或 "sem_c_"（截面）
    返回:
        字典，键是带前缀的特征名（如 "sem_s_gray_mean"），值是平均后的数值
    """
    all_features = []                           # 空列表，存放每张图片的特征字典
    success_count = 0                           # 成功处理的图片数量

    for filename in sem_files:                  # 遍历该样品的每张图片
        filepath = os.path.join(config.SEM_DIR, filename)  # 拼接完整文件路径
        feat = extract_sem_single_image(filepath)  # 从这张图片提取特征

        if feat is not None:                    # 如果提取成功（图片读取正常）
            all_features.append(feat)           # 加入列表
            success_count += 1                  # 计数器加1
        else:
            print(f"  警告: 无法读取图片 {filepath}")  # 打印警告

    if len(all_features) == 0:                  # 如果所有图片都失败了
        return None                             # 返回None

    # 把所有图片的特征取平均
    # 获取特征名列表（第一张图片的特征名即可，所有图片的特征名应该一样）
    feature_keys = list(all_features[0].keys())  # 取第一张图片的所有特征名

    averaged_features = {}                      # 存放平均后的特征
    for key in feature_keys:                    # 遍历每个特征
        values = [f[key] for f in all_features]  # 收集所有图片中这个特征的值
        averaged_features[f"{feature_prefix}{key}"] = np.mean(values)  # 计算平均值，加上前缀作为新特征名

    return averaged_features                    # 返回平均后的特征字典


# ======================== 主程序 ========================
if __name__ == "__main__":                      # 当直接运行这个脚本时执行以下代码
    print("=" * 60)                              # 打印分隔线（60个等号）
    print("步骤1: 提取特征")                     # 打印当前步骤
    print("=" * 60)                              # 打印分隔线

    # 发现所有样品并检查数据可用性
    sample_info = discover_samples()            # 调用样品发现函数

    if len(sample_info) == 0:                   # 如果没有找到样品
        print("程序终止。请放入数据文件和标签文件后再运行。")  # 提示用户
        exit()                                  # 退出程序

    # 打印数据可用性统计
    print_data_availability(sample_info)        # 打印每个数据源的可用情况

    # 检查是否所有样品都有标签
    missing_labels = []                         # 空列表，存放没有标签的样品
    for sample in sample_info:                  # 遍历每个样品（sample_info的键就是样品名）
        # 标签文件已经在discover_samples中读过了，这里只需要检查
        pass

    # 读取标签文件获取标签映射
    labels_df = pd.read_csv(config.LABEL_FILE)  # 读取标签CSV
    label_map = dict(zip(labels_df["sample_name"], labels_df["label"]))  # 转成字典: 样品名→老化级别

    # 对每个样品提取特征
    print(f"\n开始提取 {len(sample_info)} 个样品的特征...")  # 打印进度提示
    all_features = []                           # 空列表，存放所有样品的特征

    for sample, info in sample_info.items():    # 遍历每个样品（sample是名字，info是数据可用性信息）
        print(f"  处理: {sample}", end="")          # 打印当前处理的样品名
        # 打印该样品有哪些数据
        data_types = []                         # 存放该样品有的数据类型
        if info["has_xrd"]:
            data_types.append("XRD")            # 有XRD就加进去
        if info["has_ir"]:
            data_types.append("IR")             # 有IR就加进去
        if len(info["sem_surface_files"]) > 0:
            data_types.append(f"SEM表面({len(info['sem_surface_files'])}张)")  # 有表面SEM就加张数
        if len(info["sem_cross_files"]) > 0:
            data_types.append(f"SEM截面({len(info['sem_cross_files'])}张)")    # 有截面SEM就加张数

        print(f"  [{', '.join(data_types)}]")        # 打印该样品的数据类型

        # 初始化一个字典，存放该样品的所有特征
        combined = {"sample_name": sample}      # 第一列是样品名

        # 提取XRD特征（如果有）
        if info["has_xrd"]:                     # 如果有XRD数据
            try:                                # 尝试提取
                xrd_feat = extract_xrd_features(sample)  # 提取XRD特征
                combined.update(xrd_feat)       # 加入XRD特征
            except Exception as e:              # 如果提取失败
                print(f"    [警告] XRD提取失败: {e}")  # 打印警告
                combined.update(config.XRD_DEFAULTS)  # 用默认值填充
        else:
            combined.update(config.XRD_DEFAULTS)  # 没有XRD数据，用默认值填充0

        # 提取IR特征（如果有）
        if info["has_ir"]:                      # 如果有IR数据
            try:                                # 尝试提取
                ir_feat = extract_ir_features(sample)  # 提取IR特征
                combined.update(ir_feat)        # 加入IR特征
            except Exception as e:              # 如果提取失败
                print(f"    [警告] IR提取失败: {e}")  # 打印警告
                combined.update(config.IR_DEFAULTS)  # 用默认值填充
        else:
            combined.update(config.IR_DEFAULTS)  # 没有IR数据，用默认值填充0

        # 提取SEM表面特征（如果有）
        if len(info["sem_surface_files"]) > 0:  # 如果有表面SEM图片
            try:                                # 尝试提取
                sem_s_feat = extract_sem_for_sample(info["sem_surface_files"], "sem_s_")  # 提取表面特征
                if sem_s_feat is not None:      # 如果提取成功
                    combined.update(sem_s_feat)  # 加入表面特征
                else:
                    combined.update(config.SEM_SURFACE_DEFAULTS)  # 提取失败，用默认值填充
            except Exception as e:              # 如果提取失败
                print(f"    [警告] SEM表面提取失败: {e}")  # 打印警告
                combined.update(config.SEM_SURFACE_DEFAULTS)  # 用默认值填充
        else:
            combined.update(config.SEM_SURFACE_DEFAULTS)  # 没有表面SEM，用默认值填充0

        # 提取SEM截面特征（如果有）
        if len(info["sem_cross_files"]) > 0:    # 如果有截面SEM图片
            try:                                # 尝试提取
                sem_c_feat = extract_sem_for_sample(info["sem_cross_files"], "sem_c_")  # 提取截面特征
                if sem_c_feat is not None:      # 如果提取成功
                    combined.update(sem_c_feat)  # 加入截面特征
                else:
                    combined.update(config.SEM_CROSS_DEFAULTS)  # 提取失败，用默认值填充
            except Exception as e:              # 如果提取失败
                print(f"    [警告] SEM截面提取失败: {e}")  # 打印警告
                combined.update(config.SEM_CROSS_DEFAULTS)  # 用默认值填充
        else:
            combined.update(config.SEM_CROSS_DEFAULTS)  # 没有截面SEM，用默认值填充0

        # 加入老化级别标签
        combined["label"] = label_map.get(sample, "未知")  # 从标签字典中获取老化级别，没有就填"未知"

        all_features.append(combined)           # 把这个样品的特征加入总列表

    # 把特征列表转成DataFrame（表格），然后保存为CSV
    df = pd.DataFrame(all_features)             # 列表转成pandas数据表
    df.to_csv(config.FEATURES_FILE, index=False, encoding="utf-8-sig")  # 保存为CSV，index=False不要行号，utf-8-sig支持中文

    # 打印统计信息
    print(f"\n特征提取完成!")                     # 打印完成提示
    print(f"样品数量: {len(df)}")                 # 打印样品数
    print(f"特征数量: {len(df.columns) - 2}")     # 打印特征数（减去sample_name和label两列）
    print(f"特征表已保存到: {config.FEATURES_FILE}")  # 打印保存路径

    # 打印每个老化级别有多少样品（检查类别分布是否均衡）
    print("\n各老化级别样品数量:")                # 打印标题
    label_counts = df["label"].value_counts()   # 统计每个标签的数量
    for label, count in label_counts.items():   # 遍历每个标签和数量
        print(f"  {label}: {count}个")            # 打印

    # 如果有某个级别样品太少，给出警告
    if label_counts.min() < 3:                  # 如果最少的那一类不到3个样品
        print("\n[警告] 某个老化级别的样品太少（<3个），可能影响模型效果")  # 打印警告
        print("  建议: 每个级别至少5个样品")        # 给出建议

    print(f"\n下一步: 运行 'python step2_训练模型.py' 开始训练模型")  # 告诉用户下一步做什么
