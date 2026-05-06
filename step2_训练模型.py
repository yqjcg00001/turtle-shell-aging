# ============================================================
# step2_训练模型.py —— 训练分类模型 + 评估效果
# ============================================================
# 这一步做什么：
#   1. 读取 step1 生成的 features.csv（特征表）
#   2. 用 SVM（支持向量机）算法训练分类模型
#   3. 用"交叉验证"评估模型的准确率（确保不是死记硬背）
#   4. 保存训练好的模型文件（后续预测新样品要用）
#   5. 生成混淆矩阵图片（看哪些类别容易被搞混）
#
# 你需要做的：
#   1. 确保已经运行过 step1_提取特征.py，生成了 features.csv
#   2. 运行: python step2_训练模型.py
#
# 运行后生成:
#   models/svm_model.joblib   — 训练好的模型文件
#   models/scaler.joblib      — 数据归一化工具（预测新样品时要用同一个）
#   models/selector.joblib    — 特征选择器（预测新样品时要用同一个）
#   models/selected_features.txt  — 最终选用了哪些特征
#   results/confusion_matrix.png  — 混淆矩阵图片
#   results/classification_report.txt  — 详细评估报告
# ============================================================

# ======================== 导入需要的库 ========================
import os                                     # 操作系统接口，用来处理文件路径
import numpy as np                            # 数值计算库，做数学运算
import pandas as pd                           # 表格处理库，读取CSV和处理数据
import joblib                                 # 模型保存和加载工具，把训练好的模型存到硬盘上
from sklearn.svm import SVC                   # SVM分类器（支持向量机），我们的核心算法
from sklearn.preprocessing import RobustScaler  # 归一化工具（用中位数和四分位距缩放数据）
from sklearn.feature_selection import VarianceThreshold  # 方差过滤器，去掉几乎不变的特征
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # 互信息选择器，选出和分类最相关的特征
from sklearn.feature_selection import RFECV   # 递归特征消除（自动找最优特征数量）
from sklearn.model_selection import StratifiedKFold  # 分层交叉验证（保证每折的类别比例一致）
from sklearn.model_selection import cross_validate  # 交叉验证工具，自动轮流划分训练集和测试集
from sklearn.metrics import confusion_matrix, classification_report  # 评估指标：混淆矩阵和分类报告
import matplotlib                             # 画图工具
matplotlib.use("Agg")                         # 设置后端为"Agg"（不弹窗口，直接保存图片到文件）
import matplotlib.pyplot as plt               # matplotlib的画图接口
import seaborn as sns                         # seaborn是matplotlib的美化工具，画热力图很好看
import config                                 # 导入配置文件


# ======================== 读取数据 ========================
print("=" * 60)                                # 打印分隔线
print("步骤2: 训练模型")                       # 打印当前步骤
print("=" * 60)                                # 打印分隔线

# 检查特征文件是否存在
if not os.path.exists(config.FEATURES_FILE):  # 如果特征文件不存在
    print(f"\n错误: 找不到特征文件 {config.FEATURES_FILE}")  # 打印错误信息
    print("请先运行 'python step1_提取特征.py' 生成特征文件。")  # 告诉用户怎么办
    exit()                                     # 退出程序

# 读取特征表
df = pd.read_csv(config.FEATURES_FILE)         # 读取features.csv
print(f"\n读取特征表: {len(df)}个样品, {len(df.columns)}列")  # 打印基本信息

# 去掉没有标签的样品（"未知"的样品不能用来训练）
df = df[df["label"] != "未知"].copy()          # 只保留label不是"未知"的行，.copy()避免修改原始数据时的警告
print(f"有效样品（有标签的）: {len(df)}个")          # 打印有效样品数

if len(df) < 5:                                # 如果样品太少（少于5个）
    print("\n错误: 样品太少（至少需要5个有标签的样品），无法训练模型。")  # 打印错误
    exit()                                     # 退出程序

# 分离特征（X）和标签（y）
# X 是输入数据（各种特征指标），y 是输出标签（老化级别）
# 类比：X = 体检报告上的各项指标，y = 医生给出的诊断结果
X = df.drop(columns=["sample_name", "label"])  # 去掉样品名和标签列，剩下的都是特征
y = df["label"]                                # 标签列就是我们要预测的目标
feature_names = X.columns.tolist()             # 记住每个特征的名字，后面画图要用
X = X.values                                   # 把DataFrame转成numpy数组（模型需要数组格式）

# 把标签转成数字（sklearn的模型只能处理数字标签）
# LabelEncoder: 把文字标签转成数字，如 "轻度老化"→0, "中度老化"→1
from sklearn.preprocessing import LabelEncoder  # 导入标签编码工具
le = LabelEncoder()                            # 创建编码器
y_encoded = le.fit_transform(y)                # 把文字标签转成数字 [0, 1, 2, 3, ...]
class_names = [str(c) for c in le.classes_]    # 把标签转成字符串，兼容纯数字标签和文字标签
print(f"老化级别: {', '.join(class_names)}")          # 打印有哪些老化级别
print(f"各级别样品数:")                                 # 打印各级别数量
for cls, display_name in zip(le.classes_, class_names):  # 遍历每个类别
    count = sum(y == cls)                      # 统计该类别的样品数
    print(f"  {display_name}: {count}个")                     # 打印


# ======================== 特征选择 ========================
# 为什么要做特征选择？
#   假设你提取了80个特征，但只有10个真正对区分老化级别有用
#   留着没用的特征会让模型"分心"，而且容易过拟合（死记硬背）
#   所以我们要"百里挑一"，只留下最有用的特征
#
# 我们用三步筛选：
#   第1步：方差过滤 —— 去掉那些"几乎不变"的特征（没有区分能力）
#   第2步：互信息选择 —— 挑出和老化级别最相关的前30个特征
#   第3步：递归特征消除 —— 从30个里进一步筛选到最优数量（交叉验证自动确定）

print("\n--- 特征选择 ---")                     # 打印分隔提示

# 第1步：方差过滤
# 方差小的特征 = 所有样品的值都差不多 = 对分类没有帮助
var_thresh = VarianceThreshold(threshold=0.01)  # 创建方差过滤器，阈值0.01（方差<0.01的特征会被去掉）
X_var = var_thresh.fit_transform(X)            # 应用方差过滤，只保留方差够大的特征
# 记住哪些特征被保留了
var_mask = var_thresh.get_support()            # 获取保留/删除的掩码（True=保留，False=删除）
selected_names_1 = [name for name, keep in zip(feature_names, var_mask) if keep]  # 只保留被选中的特征名
removed_count = len(feature_names) - len(selected_names_1)  # 计算去掉了多少个特征
print(f"  方差过滤: {len(feature_names)}个 → {len(selected_names_1)}个 (去掉了{removed_count}个几乎不变的特征)")

# 第2步：互信息选择
# 互信息衡量"某个特征和老化级别之间的关联程度"——关联度越高，这个特征越有用
# SelectKBest: 选出得分最高的K个特征
if len(selected_names_1) > 30:                 # 如果特征还太多（超过30个）
    k = min(30, len(selected_names_1))         # 选30个（如果不够30个就全选）
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)  # 创建互信息选择器，选前k个
    X_mi = mi_selector.fit_transform(X_var, y_encoded)  # 应用互信息选择
    mi_mask = mi_selector.get_support()        # 获取选中的掩码
    selected_names_2 = [name for name, keep in zip(selected_names_1, mi_mask) if keep]  # 保留选中的特征名
    print(f"  互信息选择: {len(selected_names_1)}个 → {len(selected_names_2)}个")  # 打印结果
else:
    # 特征已经够少了，不需要再做互信息选择
    X_mi = X_var                               # 直接用方差过滤后的数据
    selected_names_2 = selected_names_1         # 特征名不变
    print(f"  互信息选择: 特征数({len(selected_names_1)})已较少，跳过此步")

# 第3步：递归特征消除（RFECV）
# 这是最关键的一步：自动找到"用几个特征效果最好"
# 原理：先用所有特征训练模型，然后一个一个去掉最不重要的特征，
#      每去掉一个就评估一次，找到性能最好的那个特征数量
print(f"  递归特征消除(RFECV): 正在寻找最优特征组合...")  # 提示用户正在进行

# 创建内层交叉验证器（用来评估每一组特征的表现）
inner_cv = StratifiedKFold(n_splits=min(3, len(df)), shuffle=True, random_state=config.RANDOM_SEED)  # 3折分层交叉验证

from sklearn.linear_model import LogisticRegression  # 逻辑回归（用于RFECV特征选择）

print(f"  递归特征消除(RFECV): 正在寻找最优特征组合...")  # 提示用户正在进行

# 创建内层交叉验证器（用来评估每一组特征的表现）
inner_cv = StratifiedKFold(n_splits=min(3, len(df)), shuffle=True, random_state=config.RANDOM_SEED)  # 3折分层交叉验证

# 创建逻辑回归分类器（用于RFECV评估每组特征）
# 用逻辑回归是因为它有coef_属性（特征权重），RFECV需要这个来判断哪些特征重要
# 注意：最终训练用的还是SVM，这里只用逻辑回归来选特征
lr_for_rfe = LogisticRegression(               # 创建逻辑回归分类器
    max_iter=1000,                             # 最大迭代次数（避免不收敛）
    random_state=config.RANDOM_SEED,           # 固定随机种子
    class_weight="balanced"                    # 自动平衡类别权重
)

# 创建RFECV（递归特征消除 + 交叉验证）
# step=1: 每次去掉1个最不重要的特征
# scoring="accuracy": 用准确率来评估每组特征的好坏
rfe = RFECV(                                   # 创建RFECV对象
    estimator=lr_for_rfe,                      # 用逻辑回归来评估特征重要性
    step=1,                                    # 每次去掉1个特征
    cv=inner_cv,                               # 用3折交叉验证评估
    scoring="accuracy",                        # 评估指标是准确率
    min_features_to_select=3                   # 最少保留3个特征（不能再少了）
)
rfe.fit(X_mi, y_encoded)                       # 运行递归特征消除

# 获取RFECV选择的结果
rfe_mask = rfe.support_                        # 获取RFECV选中的特征掩码
selected_names_final = [name for name, keep in zip(selected_names_2, rfe_mask) if keep]  # 保留最终选中的特征名
print(f"  RFECV结果: {len(selected_names_2)}个 → {len(selected_names_final)}个")  # 打印筛选结果
print(f"  最终选用的特征: {', '.join(selected_names_final)}")  # 打印最终选用了哪些特征

# 用最终选中的特征构建数据集
X_selected = X_mi[:, rfe_mask]                # 只保留被选中的特征列

# 把特征选择的结果打印出来（方便你了解哪些特征最重要）
# rfe.ranking_ 越小表示特征越重要（1表示最优）
ranking = rfe.ranking_                        # 获取特征排名
importance_order = sorted(zip(selected_names_2, ranking), key=lambda x: x[1])  # 按排名从小到大排序
print("\n  特征重要性排名（前10个）:")              # 打印标题
for i, (name, rank) in enumerate(importance_order[:10]):  # 只打印前10个
    print(f"    {i+1}. {name} (排名: {rank})")         # 打印排名


# ======================== 数据归一化 ========================
# 为什么要归一化？
#   不同特征的数值范围差很大！比如：
#     XRD强度可能是 10000，IR吸光度是 0.5，纹理特征是 0.1
#   如果不归一化，数值大的特征会"主导"模型判断，这不公平
#   归一化就是把所有特征缩放到差不多的范围
#
# 为什么用 RobustScaler 而不是 StandardScaler？
#   RobustScaler 用中位数和四分位距来缩放，对"异常值"不敏感
#   小样本数据中异常值很常见，RobustScaler 更稳定

print("\n--- 数据归一化 ---")                    # 打印分隔提示
scaler = RobustScaler()                        # 创建归一化工具（RobustScaler）
X_scaled = scaler.fit_transform(X_selected)    # 拟合并转换数据（计算中位数和四分位距，然后缩放）


# ======================== 交叉验证评估 ========================
# 什么是交叉验证？
#   把数据分成K份（比如5份），轮流用其中4份训练、1份测试
#   这样每份数据都有机会被"考"一次，结果更可靠
#
# 为什么要用交叉验证而不是直接训练？
#   如果直接用全部数据训练，模型会"记住"这些样品（过拟合）
#   交叉验证模拟了"遇到没见过的新样品"的情况，评估更真实

print("\n--- 交叉验证评估 ---")                   # 打印分隔提示

# 创建分层K折交叉验证器
# stratify: 保证每折中各类别比例和总体一致（避免某折全是"轻度"、某折全是"重度"）
outer_cv = StratifiedKFold(n_splits=min(config.CV_FOLDS, len(df)), shuffle=True, random_state=config.RANDOM_SEED)  # 分层K折

# 创建最终用于交叉验证的SVM分类器
# class_weight="balanced": 自动处理类别不平衡问题（某类样品少就给更大权重）
svm_model = SVC(                               # 创建SVM分类器
    kernel="rbf",                              # RBF核函数（径向基函数，能处理非线性关系）
    C=config.SVM_C,                            # 正则化参数，从config读取
    gamma=config.SVM_GAMMA,                    # 核函数参数，"scale"表示自动计算
    class_weight="balanced",                   # 自动平衡各类别的权重（样品少的类别权重更大）
    random_state=config.RANDOM_SEED,           # 固定随机种子，保证结果可重复
    probability=True                           # 开启概率输出（可以知道模型有多大把握）
)

# 运行交叉验证
# cv: 交叉验证器，自动划分训练集和测试集
# scoring: 要评估哪些指标
# return_train_score: 同时返回训练集分数（用来判断是否过拟合）
cv_results = cross_validate(                   # 运行交叉验证
    svm_model,                                 # 要评估的模型
    X_scaled,                                  # 特征数据
    y_encoded,                                 # 标签
    cv=outer_cv,                               # 交叉验证器
    scoring=["accuracy", "f1_macro", "recall_macro"],  # 评估指标：准确率、宏F1、宏召回
    return_train_score=True                    # 同时返回训练集得分
)

# 打印交叉验证结果
print(f"\n  交叉验证结果 ({min(config.CV_FOLDS, len(df))}折):")  # 打印标题和折数
print(f"  准确率:   {cv_results['test_accuracy'].mean():.1%} ± {cv_results['test_accuracy'].std():.1%}")  # 准确率（百分比格式）
print(f"  宏F1分数: {cv_results['test_f1_macro'].mean():.1%} ± {cv_results['test_f1_macro'].std():.1%}")  # F1分数（综合精度和召回率）
print(f"  宏召回率: {cv_results['test_recall_macro'].mean():.1%} ± {cv_results['test_recall_macro'].std():.1%}")  # 召回率（正确识别出的比例）

# 检查是否过拟合
train_acc = cv_results["train_accuracy"].mean()  # 训练集平均准确率
test_acc = cv_results["test_accuracy"].mean()    # 测试集平均准确率
gap = train_acc - test_acc                       # 训练和测试的差距
print(f"\n  训练集准确率: {train_acc:.1%}")              # 打印训练集准确率
print(f"  测试集准确率: {test_acc:.1%}")               # 打印测试集准确率
if gap > 0.2:                                    # 如果差距超过20%
    print(f"  [警告] 训练集和测试集差距较大({gap:.1%})，可能存在过拟合")  # 警告过拟合
    print(f"    建议: 减少特征数量或增大SVM的C值（在config.py中修改）")      # 给出建议
elif gap > 0.1:                                  # 如果差距超过10%
    print(f"  [注意] 训练集和测试集差距适中({gap:.1%})，模型略有过拟合倾向")  # 提示轻微过拟合
else:
    print(f"  [OK] 训练集和测试集差距较小({gap:.1%})，模型状态良好")             # 模型正常


# ======================== 训练最终模型 ========================
# 交叉验证只是"模拟考试"，现在用全部数据"正式学习"一遍
# 这样训练出来的模型能利用到所有样品的信息，效果最好

print("\n--- 训练最终模型 ---")                    # 打印分隔提示
svm_model.fit(X_scaled, y_encoded)             # 用全部数据训练SVM模型
print("模型训练完成!")                             # 打印完成提示

# 在全部训练数据上评估最终表现（仅作参考，不如交叉验证可靠）
train_pred = svm_model.predict(X_scaled)        # 用模型预测训练数据的标签
train_acc_full = np.mean(train_pred == y_encoded)  # 计算训练集准确率
print(f"训练集准确率（全部数据）: {train_acc_full:.1%}")  # 打印


# ======================== 保存模型和工具 ========================
# 为什么要保存三个东西？
#   1. svm_model: 训练好的"判断规则"
#   2. scaler: 归一化工具（预测新样品时必须用同一个归一化标准）
#   3. selector: 特征选择器（预测新样品时必须选择同样的特征）
#   如果只保存模型，预测新样品时会因为特征不匹配而报错

print("\n--- 保存模型 ---")                       # 打印分隔提示

# 保存SVM模型
model_path = os.path.join(config.MODEL_DIR, "svm_model.joblib")  # 模型文件路径
joblib.dump(svm_model, model_path)              # 保存模型到硬盘
print(f"  模型已保存至: {model_path}")              # 打印保存路径

# 保存归一化工具
scaler_path = os.path.join(config.MODEL_DIR, "scaler.joblib")  # 归一化工具路径
joblib.dump(scaler, scaler_path)                # 保存归一化工具到硬盘
print(f"  归一化工具已保存至: {scaler_path}")        # 打印保存路径

# 保存特征选择器
# 我们需要保存：方差过滤器、互信息选择器、RFECV选择器，以及对应的特征名
selector_info = {                               # 用一个字典打包所有选择器信息
    "var_thresh": var_thresh,                   # 方差过滤器
    "mi_selector": mi_selector if len(feature_names) > 30 else None,  # 互信息选择器（如果没有用就是None）
    "rfe": rfe,                                 # RFECV选择器
    "feature_names_all": feature_names,         # 训练时的全部特征名列表（预测时需要按这个顺序排列）
    "feature_names_after_var": selected_names_1,  # 方差过滤后的特征名列表
    "feature_names_after_mi": selected_names_2,  # 互信息选择后的特征名列表
    "feature_names_final": selected_names_final  # 最终选用的特征名列表
}
selector_path = os.path.join(config.MODEL_DIR, "selector.joblib")  # 选择器文件路径
joblib.dump(selector_info, selector_path)       # 保存选择器信息到硬盘
print(f"  特征选择器已保存至: {selector_path}")      # 打印保存路径

# 保存标签编码器（把数字标签转回文字要用）
encoder_path = os.path.join(config.MODEL_DIR, "label_encoder.joblib")  # 编码器路径
joblib.dump(le, encoder_path)                   # 保存标签编码器
print(f"  标签编码器已保存至: {encoder_path}")       # 打印保存路径

# 保存特征列表（方便查看）
features_txt = os.path.join(config.MODEL_DIR, "selected_features.txt")  # 特征列表文件路径
with open(features_txt, "w", encoding="utf-8") as f:  # 打开文件准备写入
    f.write("最终选用的特征:\n")                      # 写入标题
    for name in selected_names_final:            # 遍历每个最终特征
        f.write(f"  - {name}\n")                  # 写入特征名
print(f"  特征列表已保存至: {features_txt}")        # 打印保存路径


# ======================== 生成混淆矩阵图片 ========================
# 混淆矩阵是什么？
#   一个表格，行=实际类别，列=预测类别
#   对角线上的数字 = 预测正确的数量
#   非对角线上的数字 = 预测错误的数量（哪个格子大，说明哪两类容易搞混）

print("\n--- 生成混淆矩阵 ---")                    # 打印分隔提示

# 计算混淆矩阵
cm = confusion_matrix(y_encoded, train_pred)     # 计算混淆矩阵（实际标签 vs 预测标签）

# 画图
fig, ax = plt.subplots(figsize=(8, 6))          # 创建画布，大小8x6英寸
# 用seaborn画热力图（颜色深浅表示数量多少）
sns.heatmap(                                     # 热力图函数
    cm,                                          # 混淆矩阵数据
    annot=True,                                  # 在格子里显示数字
    fmt="d",                                     # 数字格式：整数(d)
    cmap="Blues",                                # 颜色方案：蓝色系
    xticklabels=class_names,                     # X轴标签：类别名
    yticklabels=class_names                      # Y轴标签：类别名
)
ax.set_xlabel("预测标签", fontsize=12)            # X轴标题
ax.set_ylabel("实际标签", fontsize=12)            # Y轴标题
ax.set_title("混淆矩阵（训练集）", fontsize=14)    # 图标题
plt.tight_layout()                               # 自动调整布局，防止标签被截断
cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")  # 保存路径
plt.savefig(cm_path, dpi=150)                   # 保存图片，分辨率150
plt.close()                                     # 关闭画布（释放内存）
print(f"  混淆矩阵已保存至: {cm_path}")               # 打印保存路径

# 怎么看混淆矩阵？
#   1. 对角线上的数字越大越好（预测正确）
#   2. 如果某个非对角线格子数字大，说明这两类容易搞混
#   3. 打开 results/confusion_matrix.png 就能看图


# ======================== 生成分类报告 ========================
# 分类报告详细列出每个类别的精确率、召回率、F1分数
#   精确率(Precision): 模型说"这是A类"的样品中，有多少真的是A类
#   召回率(Recall): 所有真正的A类样品中，模型找出了多少
#   F1分数: 精确率和召回率的综合（两个都高，F1才高）

print("\n--- 生成分类报告 ---")                    # 打印分隔提示
report = classification_report(                  # 生成分类报告
    y_encoded,                                   # 实际标签
    train_pred,                                  # 预测标签
    target_names=class_names                     # 类别名（让报告显示中文而不是数字）
)
report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")  # 报告保存路径
with open(report_path, "w", encoding="utf-8") as f:  # 打开文件
    f.write("分类报告（训练集）\n")                   # 写入标题
    f.write("=" * 40 + "\n")                      # 写入分隔线
    f.write(report)                               # 写入报告内容
print(f"  分类报告已保存至: {report_path}")          # 打印保存路径

# 也把报告打印到屏幕上，让你直接看到
print(f"\n{report}")                               # 打印分类报告到终端


# ======================== 完成总结 ========================
print("\n" + "=" * 60)                           # 打印分隔线
print("步骤2 完成!")                               # 打印完成提示
print("=" * 60)                                  # 打印分隔线
print("\n[结果摘要]")                             # 打印摘要标题
print(f"  样品总数: {len(df)}")                         # 打印样品数
print(f"  原始特征数: {len(feature_names)}")              # 打印原始特征数
print(f"  最终特征数: {len(selected_names_final)}")      # 打印最终特征数
print(f"  交叉验证准确率: {cv_results['test_accuracy'].mean():.1%}")  # 打印CV准确率

# 给出后续操作指引
print("\n下一步操作:")                             # 打印提示
print(f"  1. 查看 results/confusion_matrix.png 看混淆矩阵")  # 查看混淆矩阵
print(f"  2. 查看 results/classification_report.txt 看详细报告")  # 查看详细报告
print(f"  3. 运行 'python step3_预测新样品.py --xrd data/xrd/xxx_xrd.csv --ir data/ir/xxx_ir.csv --sem data/sem/xxx_sem.jpg' 预测新样品")  # 预测新样品
print(f"  4. 运行 'python step4_看看结果.py' 查看更多可视化图表")  # 查看更多图表
