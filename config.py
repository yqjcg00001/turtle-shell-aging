# ============================================================
# 配置文件 - 在这里设置所有参数
# 如果你不懂某个参数是什么意思，保持默认值就行，不用改
# ============================================================

import os                                     # 导入操作系统交互模块，用于处理文件路径


# ===== 数据路径设置 =====
# BASE_DIR: 项目根目录，自动获取当前文件所在的路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件夹路径
DATA_DIR = os.path.join(BASE_DIR, "data")                     # 数据总目录
XRD_DIR = os.path.join(DATA_DIR, "xrd")                       # XRD的CSV文件放在这里
IR_DIR = os.path.join(DATA_DIR, "ir")                         # IR的CSV文件放在这里
SEM_DIR = os.path.join(DATA_DIR, "sem")                       # SEM的JPG/TIF/PNG图片放在这里
LABEL_FILE = os.path.join(DATA_DIR, "sample_labels.csv")      # 样品标签文件（你手动填写的）

# 输出路径
FEATURES_FILE = os.path.join(BASE_DIR, "features.csv")        # 特征提取结果（自动生成）
MODEL_DIR = os.path.join(BASE_DIR, "models")                  # 训练好的模型保存位置
RESULTS_DIR = os.path.join(BASE_DIR, "results")               # 结果图片保存位置

# 确保输出文件夹存在，如果不存在就创建
os.makedirs(MODEL_DIR, exist_ok=True)       # exist_ok=True: 如果文件夹已存在就不报错
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===== 模型参数设置 =====
# 随机种子：保证每次运行结果一致（可复现）
# 就像做实验要控制变量一样，固定随机数让结果可以重复
RANDOM_SEED = 42

# 老化级别列表
# 根据你的实际分类修改，顺序对应数字 0, 1, 2, 3
# 比如你有5级，就加一个: ["1未老化", "2轻度", "3中度", "4重度", "5严重"]
AGING_CLASSES = ["1轻度老化", "2中度老化", "3重度老化", "4严重老化"]

# 交叉验证折数：把数据分成几份轮流验证
# 5折 = 用4/5的数据训练，1/5测试，轮流换5次
# 样品太少（<30）可以改成3
CV_FOLDS = 5

# SVM参数
# C值越大模型越复杂（容易过拟合），越小越简单（可能欠拟合）
SVM_C = 1.0
# gamma控制模型对单个样品的关注程度，"scale"是自动计算
SVM_GAMMA = "scale"


# ===== 特征提取参数 =====

# XRD参数
# 结晶度计算时，结晶峰区域的角度范围（2θ角度）
# 羟基磷灰石的主要结晶峰通常在 ~32° 附近，根据你的实际谱图调整
XRD_CRYSTALLINE_REGIONS = [(30.0, 35.0), (25.0, 29.0)]       # 结晶峰区域
XRD_AMORPHOUS_REGION = (15.0, 22.0)                            # 非晶（无定形）区域
XRD_PEAK_PROMINENCE = 0.1                                      # 找峰的最低显著性（越小找到的峰越多）
XRD_PEAK_DISTANCE = 5                                          # 两个峰之间最少隔几个数据点

# IR参数
# 官能团对应的波数范围（cm⁻¹），根据你的实际谱图调整
# 这些是龟甲中常见官能团的典型位置
IR_FUNCTIONAL_GROUPS = {
    "PO4磷酸根": (900, 1200),        # 磷酸根吸收区，羟基磷灰石的特征
    "CO3碳酸根": (1350, 1550),       # 碳酸根吸收区，生物磷灰石中常见
    "酰胺I": (1580, 1720),           # 酰胺I带，蛋白质（胶原蛋白）的特征
    "酰胺II": (1480, 1580),          # 酰胺II带，蛋白质特征
    "OH羟基": (3000, 3700),          # 羟基吸收区
}

# SEM参数
SEM_IMAGE_SIZE = (512, 512)          # 把SEM图片统一缩放到这个大小（宽, 高）
GLCM_DISTANCE = 1                    # 纹理分析时像素之间的计算距离
GLCM_LEVELS = 256                    # 灰度级数（256级 = 标准8位灰度图）

# SEM文件名识别：什么样的文件名算"表面"，什么样的算"截面"
# 只要文件名包含下面的关键词，就会被归到对应类别
SEM_SURFACE_KEYWORDS = ["surface", "biaomian", "bm", "表面"]        # 表面的关键词（中英文都支持）
SEM_CROSS_KEYWORDS = ["cross", "section", "jiemian", "jm", "截面"]  # 截面的关键词（中英文都支持）
SEM_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"]  # 支持的图片格式


# ===== 默认特征值字典（样品缺少某个数据源时填0） =====
# 下面这些字典定义了当某个样品缺少某种数据时，对应特征应该填什么值（默认都是0）
# 这些特征名必须和特征提取函数生成的完全一致

# XRD缺失时的默认特征值
XRD_DEFAULTS = {
    "xrd_mean_intensity": 0.0, "xrd_max_intensity": 0.0, "xrd_std_intensity": 0.0,
    "xrd_total_area": 0.0, "xrd_skewness": 0.0, "xrd_kurtosis": 0.0,
    "xrd_peak_count": 0.0,
    "xrd_peak1_position": 0.0, "xrd_peak1_height": 0.0,
    "xrd_peak2_position": 0.0, "xrd_peak2_height": 0.0,
    "xrd_peak3_position": 0.0, "xrd_peak3_height": 0.0,
    "xrd_crystallinity_index": 0.0,
    "xrd_fwhm": 0.0, "xrd_crystallite_size": 0.0
}

# IR缺失时的默认特征值
IR_DEFAULTS = {
    "ir_mean_absorbance": 0.0, "ir_max_absorbance": 0.0, "ir_std_absorbance": 0.0,
    "ir_total_area": 0.0, "ir_peak_count": 0.0,
    "ir_peak1_position": 0.0, "ir_peak1_height": 0.0,
    "ir_peak2_position": 0.0, "ir_peak2_height": 0.0,
    "ir_peak3_position": 0.0, "ir_peak3_height": 0.0,
    "ir_peak4_position": 0.0, "ir_peak4_height": 0.0,
    "ir_peak5_position": 0.0, "ir_peak5_height": 0.0,
    "ir_ratio_PO4_CO3": 0.0, "ir_ratio_amide1_amide2": 0.0
}

# SEM表面缺失时的默认特征值
SEM_SURFACE_DEFAULTS = {
    "sem_s_gray_mean": 0.0, "sem_s_gray_std": 0.0, "sem_s_gray_median": 0.0,
    "sem_s_gray_min": 0.0, "sem_s_gray_max": 0.0,
    "sem_s_gray_skewness": 0.0, "sem_s_gray_kurtosis": 0.0,
    "sem_s_glcm_contrast": 0.0, "sem_s_glcm_dissimilarity": 0.0,
    "sem_s_glcm_homogeneity": 0.0, "sem_s_glcm_energy": 0.0,
    "sem_s_glcm_correlation": 0.0,
    "sem_s_lbp_0": 0.0, "sem_s_lbp_1": 0.0, "sem_s_lbp_2": 0.0,
    "sem_s_lbp_3": 0.0, "sem_s_lbp_4": 0.0, "sem_s_lbp_5": 0.0,
    "sem_s_lbp_6": 0.0, "sem_s_lbp_7": 0.0, "sem_s_lbp_8": 0.0,
    "sem_s_lbp_9": 0.0,
    "sem_s_particle_count": 0.0,
    "sem_s_morph_mean_area": 0.0, "sem_s_morph_std_area": 0.0,
    "sem_s_morph_max_area": 0.0, "sem_s_morph_area_fraction": 0.0,
    "sem_s_morph_area_cv": 0.0
}

# SEM截面缺失时的默认特征值
SEM_CROSS_DEFAULTS = {
    "sem_c_gray_mean": 0.0, "sem_c_gray_std": 0.0, "sem_c_gray_median": 0.0,
    "sem_c_gray_min": 0.0, "sem_c_gray_max": 0.0,
    "sem_c_gray_skewness": 0.0, "sem_c_gray_kurtosis": 0.0,
    "sem_c_glcm_contrast": 0.0, "sem_c_glcm_dissimilarity": 0.0,
    "sem_c_glcm_homogeneity": 0.0, "sem_c_glcm_energy": 0.0,
    "sem_c_glcm_correlation": 0.0,
    "sem_c_lbp_0": 0.0, "sem_c_lbp_1": 0.0, "sem_c_lbp_2": 0.0,
    "sem_c_lbp_3": 0.0, "sem_c_lbp_4": 0.0, "sem_c_lbp_5": 0.0,
    "sem_c_lbp_6": 0.0, "sem_c_lbp_7": 0.0, "sem_c_lbp_8": 0.0,
    "sem_c_lbp_9": 0.0,
    "sem_c_particle_count": 0.0,
    "sem_c_morph_mean_area": 0.0, "sem_c_morph_std_area": 0.0,
    "sem_c_morph_max_area": 0.0, "sem_c_morph_area_fraction": 0.0,
    "sem_c_morph_area_cv": 0.0
}
