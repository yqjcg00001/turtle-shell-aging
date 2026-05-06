# ============================================================
# step4_看看结果.py —— 可视化查看模型效果
# ============================================================
# 这一步做什么：
#   1. 加载训练好的模型和特征数据
#   2. 生成多张可视化图表帮助你理解模型表现
#   3. 包括：混淆矩阵、特征重要性、PCA降维图、各类别样本分布等
#
# 你需要做的：
#   1. 确保已经运行过 step1_提取特征.py 和 step2_训练模型.py
#   2. 运行: python step4_看看结果.py
#
# 运行后生成:
#   results/confusion_matrix.png     — 混淆矩阵（预测对了几个、错了几个）
#   results/feature_importance.png   — 特征重要性排名（哪些指标最有用）
#   results/pca_scatter.png          — PCA降维散点图（样品在二维空间中的分布）
#   results/class_distribution.png   — 各类别样品数量柱状图
#   results/classification_report.txt — 文字版详细报告
# ============================================================

# ======================== 导入需要的库 ========================
import os                                     # 操作系统接口，处理文件路径
import numpy as np                            # 数值计算库
import pandas as pd                           # 表格处理库
import joblib                                 # 模型加载工具
import matplotlib                             # 画图工具
matplotlib.use("Agg")                         # 不弹窗口，直接保存图片
import matplotlib.pyplot as plt               # 画图接口
import seaborn as sns                         # 统计图表美化
from sklearn.metrics import confusion_matrix, classification_report  # 评估指标
from sklearn.decomposition import PCA        # PCA降维（用来把多维数据压缩到2D画图）
from sklearn.preprocessing import RobustScaler  # 归一化工具（重新加载用于验证）
import config                                 # 导入配置文件


# ======================== 主程序 ========================
if __name__ == "__main__":                     # 直接运行此脚本时执行
    print("=" * 60)                            # 打印分隔线
    print("步骤4: 查看模型结果")                 # 打印当前步骤
    print("=" * 60)                            # 打印分隔线

    # --- 检查必要的文件是否存在 ---
    # 检查特征文件
    if not os.path.exists(config.FEATURES_FILE):  # 特征文件不存在
        print(f"\n错误: 找不到特征文件 {config.FEATURES_FILE}")  # 打印错误
        print("请先运行 'python step1_提取特征.py'")  # 告诉用户先运行step1
        exit()                                 # 退出

    # 检查模型文件
    model_path = os.path.join(config.MODEL_DIR, "svm_model.joblib")  # 模型路径
    if not os.path.exists(model_path):         # 模型不存在
        print(f"\n错误: 找不到模型文件 {model_path}")  # 打印错误
        print("请先运行 'python step2_训练模型.py'")  # 告诉用户先运行step2
        exit()                                 # 退出

    # --- 加载数据 ---
    print("\n加载数据和模型...")                  # 提示
    df = pd.read_csv(config.FEATURES_FILE)     # 读取特征表
    df = df[df["label"] != "未知"].copy()       # 只保留有标签的样品
    print(f"  加载了 {len(df)} 个有标签的样品")       # 打印样品数

    # 加载模型和工具
    model = joblib.load(model_path)            # 加载SVM模型
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.joblib")  # 归一化工具路径
    scaler = joblib.load(scaler_path)          # 加载归一化工具
    selector_path = os.path.join(config.MODEL_DIR, "selector.joblib")  # 选择器路径
    selector_info = joblib.load(selector_path)  # 加载选择器信息
    encoder_path = os.path.join(config.MODEL_DIR, "label_encoder.joblib")  # 编码器路径
    le = joblib.load(encoder_path)             # 加载标签编码器

    # 准备特征数据
    X = df.drop(columns=["sample_name", "label"])  # 去掉非特征列
    y = df["label"]                            # 标签列
    feature_names_all = X.columns.tolist()     # CSV中全部特征名列表

    # 按照训练时的特征顺序提取特征
    X_array = X.values                         # 直接取全部特征（CSV就是训练时的数据）

    # 应用特征选择（和训练时一样的流程）
    X_var = selector_info["var_thresh"].transform(X_array)  # 方差过滤
    if selector_info["mi_selector"] is not None:  # 如果用了互信息选择
        X_mi = selector_info["mi_selector"].transform(X_var)  # 互信息选择
    else:
        X_mi = X_var                           # 没用就跳过
    X_selected = selector_info["rfe"].transform(X_mi)  # RFECV选择
    X_scaled = scaler.transform(X_selected)    # 归一化

    # 用模型预测训练数据（用于展示混淆矩阵等）
    y_encoded = le.transform(y)                # 把文字标签转成数字
    y_pred = model.predict(X_scaled)           # 预测结果

    # 把类别名转成字符串（兼容纯数字标签）
    class_names = [str(c) for c in le.classes_]

    # ======================== 图1: 混淆矩阵 ========================
    print("\n--- 生成混淆矩阵图 ---")              # 提示
    cm = confusion_matrix(y_encoded, y_pred)   # 计算混淆矩阵

    fig, ax = plt.subplots(figsize=(8, 6))     # 创建画布
    # 用seaborn画热力图，annot=True显示数字，fmt='d'整数格式
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,       # X轴：预测的类别
                yticklabels=class_names)       # Y轴：实际的类别
    ax.set_xlabel("预测标签", fontsize=12)      # X轴标题
    ax.set_ylabel("实际标签", fontsize=12)      # Y轴标题
    ax.set_title("混淆矩阵 — 对角线=预测正确，非对角线=预测错误", fontsize=14)  # 图标题
    plt.tight_layout()                         # 自动调整布局

    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")  # 保存路径
    plt.savefig(cm_path, dpi=150)              # 保存图片，分辨率150
    plt.close()                                # 关闭画布（释放内存）
    print(f"  混淆矩阵已保存至: {cm_path}")           # 打印保存路径
    print("  怎么看：对角线上是预测对的，数字越大越好；非对角线上是预测错的")  # 简单解释

    # ======================== 图2: 特征重要性 ========================
    print("\n--- 生成特征重要性图 ---")             # 提示
    rfe = selector_info["rfe"]                 # 获取RFECV选择器
    final_feature_names = selector_info["feature_names_final"]  # 最终选中的特征名
    mi_feature_names = selector_info["feature_names_after_mi"]  # 进入RFECV的特征名

    # RFECV的ranking_: 1=最好，数字越大越不重要
    # 从全部排名中提取最终选中特征的排名
    all_rankings = rfe.ranking_                # 全部特征（MI筛选后的）的排名
    # 找到最终特征在MI特征列表中的位置
    final_rankings = []
    for name in final_feature_names:           # 遍历每个最终特征
        if name in mi_feature_names:           # 如果在MI特征列表中
            idx = mi_feature_names.index(name)  # 找到它在MI列表中的位置
            final_rankings.append(all_rankings[idx])  # 获取排名
        else:
            final_rankings.append(1)           # 找不到就设为1（最好）

    fig, ax = plt.subplots(figsize=(10, max(6, len(final_feature_names) * 0.35)))  # 创建画布
    y_pos = range(len(final_feature_names))    # Y轴位置
    bars = ax.barh(y_pos, final_rankings, color="steelblue")  # 水平条形图

    # 反转Y轴，让排名最好的（值最小的）在最上面
    ax.invert_yaxis()                          # 反转Y轴
    ax.set_yticks(y_pos)                       # 设置Y轴刻度
    ax.set_yticklabels(final_feature_names, fontsize=10)  # 设置Y轴标签为特征名
    ax.set_xlabel("RFECV排名（1=最重要，数字越大越不重要）", fontsize=12)  # X轴标题
    ax.set_title("特征重要性排名（SVM-RFECV）", fontsize=14)  # 图标题
    plt.tight_layout()                         # 自动调整布局

    fi_path = os.path.join(config.RESULTS_DIR, "feature_importance.png")  # 保存路径
    plt.savefig(fi_path, dpi=150)              # 保存图片
    plt.close()                                # 关闭画布
    print(f"  特征重要性图已保存至: {fi_path}")      # 打印保存路径
    print("  怎么看：排名=1的特征最有区分能力，排名越高越有用")  # 简单解释

    # ======================== 图3: PCA降维散点图 ========================
    print("\n--- 生成PCA散点图 ---")               # 提示
    # PCA: 把多维数据压缩到2维，方便可视化
    # 可以看到不同老化级别的样品在特征空间中是否分开

    # 如果特征数超过2个，才做PCA降维
    if X_scaled.shape[1] > 2:                  # 特征数大于2才需要降维
        pca = PCA(n_components=2)              # 创建PCA对象，降到2维
        X_pca = pca.fit_transform(X_scaled)    # 拟合并转换数据
    else:
        X_pca = X_scaled                       # 如果特征本身就<=2个，直接用

    fig, ax = plt.subplots(figsize=(10, 7))    # 创建画布
    # 为每个类别分配不同的颜色和标记
    colors = plt.cm.Set1.colors                # 使用Set1配色方案
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]  # 不同的标记符号

    for i, cls in enumerate(le.classes_):      # 遍历每个类别
        # 找到属于这个类别的样品索引
        mask = y_encoded == i                  # 布尔掩码
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],  # 画散点：PCA第1维 vs 第2维
                   c=[colors[i % len(colors)]],  # 颜色（循环使用）
                   marker=markers[i % len(markers)],  # 标记（循环使用）
                   label=cls,                  # 图例标签
                   s=80,                       # 点的大小
                   alpha=0.7,                  # 透明度
                   edgecolors="black",         # 边框颜色
                   linewidths=0.5)             # 边框宽度

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差解释)" if X_scaled.shape[1] > 2 else "特征1", fontsize=12)  # X轴标签
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差解释)" if X_scaled.shape[1] > 2 else "特征2", fontsize=12)  # Y轴标签
    ax.set_title("PCA降维散点图 — 不同老化级别的样品分布", fontsize=14)  # 图标题
    ax.legend(fontsize=11)                     # 显示图例
    ax.grid(True, alpha=0.3)                   # 显示网格

    plt.tight_layout()                         # 自动调整布局
    pca_path = os.path.join(config.RESULTS_DIR, "pca_scatter.png")  # 保存路径
    plt.savefig(pca_path, dpi=150)             # 保存图片
    plt.close()                                # 关闭画布
    print(f"  PCA散点图已保存至: {pca_path}")       # 打印保存路径
    print("  怎么看：不同颜色的点越分开，说明模型越容易区分这些类别")  # 简单解释

    # ======================== 图4: 类别分布柱状图 ========================
    print("\n--- 生成类别分布图 ---")              # 提示
    label_counts = y.value_counts()            # 统计每个类别的样品数量

    fig, ax = plt.subplots(figsize=(8, 5))     # 创建画布
    bars = ax.bar(label_counts.index, label_counts.values,  # 画柱状图
                   color=sns.color_palette("pastel"),  # 柔和的配色
                   edgecolor="black",           # 柱子边框黑色
                   linewidth=0.5)               # 边框宽度

    # 在每个柱子上标注数量
    for bar, count in zip(bars, label_counts.values):  # 遍历每个柱子和数量
        height = bar.get_height()              # 柱子高度
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,  # 文字位置（柱顶上方）
                f'{int(count)}',               # 文字内容（数量）
                ha='center',                   # 水平居中
                va='bottom',                   # 垂直对齐
                fontsize=12,                   # 字体大小
                fontweight='bold')             # 加粗

    ax.set_xlabel("老化级别", fontsize=12)      # X轴标题
    ax.set_ylabel("样品数量", fontsize=12)      # Y轴标题
    ax.set_title("各老化级别样品数量分布", fontsize=14)  # 图标题
    ax.set_ylim(0, max(label_counts.values) * 1.2)  # Y轴范围（留点空间给标签）
    plt.tight_layout()                         # 自动调整布局

    cd_path = os.path.join(config.RESULTS_DIR, "class_distribution.png")  # 保存路径
    plt.savefig(cd_path, dpi=150)              # 保存图片
    plt.close()                                # 关闭画布
    print(f"  类别分布图已保存至: {cd_path}")      # 打印保存路径

    # 检查是否有类别样品太少
    if label_counts.min() < 3:                 # 最少的那一类不到3个
        print(f"\n  [注意] '{label_counts.idxmin()}' 只有 {label_counts.min()} 个样品，可能影响模型效果")  # 警告
        print("    建议: 每个类别至少5个样品效果较好")  # 建议

    # ======================== 生成文字报告 ========================
    print("\n--- 生成分类报告 ---")                # 提示
    report = classification_report(            # 生成分类报告
        y_encoded,                             # 实际标签
        y_pred,                                # 预测标签
        target_names=class_names               # 类别名（已转成字符串）
    )
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")  # 报告保存路径
    with open(report_path, "w", encoding="utf-8") as f:  # 打开文件
        f.write("分类报告（训练集）\n")               # 写入标题
        f.write("=" * 40 + "\n")                # 写入分隔线
        f.write(report)                         # 写入报告内容
    print(f"  分类报告已保存至: {report_path}")    # 打印保存路径

    # 也把报告打印到屏幕上，让你直接看到
    print(f"\n{report}")                         # 打印分类报告到终端

    # ======================== 完成总结 ========================
    print("\n" + "=" * 60)                    # 打印分隔线
    print("步骤4 完成!")                           # 打印完成提示
    print("=" * 60)                           # 打印分隔线
    print(f"\n结果文件都在 {config.RESULTS_DIR}/ 目录下:")  # 打印结果目录
    print(f"  1. confusion_matrix.png     — 混淆矩阵（预测对了几个、错了几个）")  # 说明每个文件
    print(f"  2. feature_importance.png   — 特征重要性（哪些指标最有用）")
    print(f"  3. pca_scatter.png           — PCA散点图（样品在二维空间中的分布）")
    print(f"  4. class_distribution.png    — 各类别样品数量分布")
    print(f"  5. classification_report.txt — 详细分类报告（精确率、召回率、F1）")
    print(f"\n下一步: 运行 'python step3_预测新样品.py --xrd <xrd文件> --ir <ir文件> --sem <sem文件>' 预测新样品")  # 告诉用户下一步
