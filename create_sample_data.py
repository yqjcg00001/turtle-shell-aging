# ============================================================
# create_sample_data.py —— 创建模拟数据，让你可以测试全流程
# ============================================================
# 这一步做什么：
#   生成一些模拟的 XRD/IR/SEM 数据，让你在没有真实数据的情况下
#   也能跑通整个流程，看看程序是怎么工作的
#
# 模拟数据包括：
#   - 32个样品，4个老化级别（各8个）
#   - SEM分为表面(surface)和截面(cross)，每个样品每种类型有1-2张图
#   - 模拟部分样品数据不全的情况（3个样品只有IR，3个只有XRD+IR，3个只有IR+SEM）
#
# 你需要做的：
#   运行: python create_sample_data.py
#
# 运行后会在 data/ 目录下生成模拟数据
# ============================================================

import os                                     # 操作系统接口
import numpy as np                            # 数值计算
import pandas as pd                           # 表格处理
import cv2                                    # OpenCV图像处理
import config                                 # 导入配置


def generate_xrd_csv(filepath, aging_level):
    """
    生成模拟的XRD CSV数据
    不同老化级别有不同的"峰"特征
    """
    angles = np.arange(10.0, 60.0, 0.1)       # 从10到60，步长0.1
    intensities = np.random.normal(100, 10, len(angles))  # 均值100，标准差10的随机数

    if "轻度" in str(aging_level):
        peak_strength = 500
        peak_width = 0.5
        for pos in [25.5, 31.8, 32.2, 45.0]:
            intensities += peak_strength * np.exp(-0.5 * ((angles - pos) / peak_width) ** 2)
    elif "中度" in str(aging_level):
        peak_strength = 350
        peak_width = 1.0
        for pos in [25.3, 31.5, 32.0, 44.5]:
            intensities += peak_strength * np.exp(-0.5 * ((angles - pos) / peak_width) ** 2)
    elif "重度" in str(aging_level):
        peak_strength = 200
        peak_width = 1.5
        for pos in [25.0, 31.0, 31.8, 44.0]:
            intensities += peak_strength * np.exp(-0.5 * ((angles - pos) / peak_width) ** 2)
    else:
        peak_strength = 100
        peak_width = 2.0
        for pos in [24.8, 30.5, 31.5, 43.5]:
            intensities += peak_strength * np.exp(-0.5 * ((angles - pos) / peak_width) ** 2)

    intensities = np.maximum(intensities, 0)  # 负值变成0

    df = pd.DataFrame({"角度(2θ)": angles, "强度": intensities})  # 创建数据表
    df.to_csv(filepath, index=False)          # 保存CSV


def generate_ir_csv(filepath, aging_level):
    """
    生成模拟的IR CSV数据
    不同老化级别有不同的吸收峰
    """
    wavenumbers = np.arange(400, 4000, 2)     # 从400到4000，步长2
    absorbance = np.random.normal(0.05, 0.02, len(wavenumbers))  # 小的随机背景

    if "轻度" in str(aging_level):
        for pos, strength in [(1650, 0.8), (1550, 0.6), (1030, 0.7), (1415, 0.3)]:
            absorbance += strength * np.exp(-0.5 * ((wavenumbers - pos) / 15) ** 2)
    elif "中度" in str(aging_level):
        for pos, strength in [(1650, 0.5), (1550, 0.4), (1030, 0.6), (1415, 0.5)]:
            absorbance += strength * np.exp(-0.5 * ((wavenumbers - pos) / 20) ** 2)
    elif "重度" in str(aging_level):
        for pos, strength in [(1650, 0.3), (1550, 0.25), (1030, 0.5), (1415, 0.7)]:
            absorbance += strength * np.exp(-0.5 * ((wavenumbers - pos) / 25) ** 2)
    else:
        for pos, strength in [(1650, 0.15), (1550, 0.1), (1030, 0.4), (1415, 0.9)]:
            absorbance += strength * np.exp(-0.5 * ((wavenumbers - pos) / 30) ** 2)

    absorbance += 0.4 * np.exp(-0.5 * ((wavenumbers - 3400) / 200) ** 2)
    absorbance = np.maximum(absorbance, 0)    # 负值变0

    df = pd.DataFrame({"波数(cm-1)": wavenumbers, "吸光度": absorbance})
    df.to_csv(filepath, index=False)


def generate_sem_image(filepath, aging_level):
    """
    生成模拟的SEM图片
    不同老化级别有不同的表面纹理
    """
    img = np.zeros((512, 512), dtype=np.uint8)  # 全黑的512x512图像

    if "轻度" in str(aging_level):
        for i in range(0, 512, 20):
            cv2.line(img, (0, i), (512, i + 30), 150, 1)
        noise = np.random.randint(0, 30, (512, 512), dtype=np.uint8)
        img = cv2.add(img, noise)
    elif "中度" in str(aging_level):
        for i in range(0, 512, 15):
            cv2.line(img, (0, i + np.random.randint(-5, 5)), (512, i + 30 + np.random.randint(-5, 5)), 120, 2)
        for _ in range(10):
            x1, y1 = np.random.randint(0, 512), np.random.randint(0, 512)
            x2, y2 = np.random.randint(0, 512), np.random.randint(0, 512)
            cv2.line(img, (x1, y1), (x2, y2), 200, 1)
        noise = np.random.randint(0, 40, (512, 512), dtype=np.uint8)
        img = cv2.add(img, noise)
    elif "重度" in str(aging_level):
        for _ in range(30):
            x1, y1 = np.random.randint(0, 512), np.random.randint(0, 512)
            x2, y2 = np.random.randint(0, 512), np.random.randint(0, 512)
            cv2.line(img, (x1, y1), (x2, y2), 180, 2)
        for _ in range(15):
            x, y = np.random.randint(0, 512), np.random.randint(0, 512)
            w, h = np.random.randint(10, 50), np.random.randint(10, 50)
            cv2.rectangle(img, (x, y), (x + w, y + h), 100, 1)
        noise = np.random.randint(0, 50, (512, 512), dtype=np.uint8)
        img = cv2.add(img, noise)
    else:
        for _ in range(50):
            x1, y1 = np.random.randint(0, 512), np.random.randint(0, 512)
            x2, y2 = np.random.randint(0, 512), np.random.randint(0, 512)
            cv2.line(img, (x1, y1), (x2, y2), 220, 3)
        for _ in range(25):
            x, y = np.random.randint(0, 512), np.random.randint(0, 512)
            w, h = np.random.randint(5, 80), np.random.randint(5, 80)
            cv2.rectangle(img, (x, y), (x + w, y + h), 80, 1)
        noise = np.random.randint(0, 60, (512, 512), dtype=np.uint8)
        img = cv2.add(img, noise)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 灰度转BGR
    cv2.imwrite(filepath, img_color)           # 保存JPG图片


# ======================== 主程序 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("创建模拟数据（支持SEM表面/截面 + 部分数据缺失）")
    print("=" * 60)

    classes = ["1轻度老化", "2中度老化", "3重度老化", "4严重老化"]

    os.makedirs(config.XRD_DIR, exist_ok=True)
    os.makedirs(config.IR_DIR, exist_ok=True)
    os.makedirs(config.SEM_DIR, exist_ok=True)

    labels = []
    sample_count = 0
    ir_only_count = 0                          # 计数器：只有IR的样品
    xrd_ir_only_count = 0                      # 计数器：只有XRD+IR的样品
    ir_sem_only_count = 0                      # 计数器：只有IR+SEM的样品
    complete_count = 0                         # 计数器：三种都有的样品

    for cls in classes:
        for i in range(8):
            sample_name = f"GH{sample_count + 1:03d}"  # 样品名：GH001, GH002, ...
            print(f"  生成: {sample_name} ({cls})")

            # 决定这个样品有哪些数据（模拟部分缺失）
            # 前3个只有IR，接下来3个XRD+IR，再3个IR+SEM，其余三种都有
            mod_idx = sample_count % 12        # 每12个循环一次
            has_xrd = True                     # 默认有XRD
            has_sem = True                     # 默认有SEM
            has_ir = True                      # 所有样品都有IR（作为保底）

            if mod_idx < 3:                    # 前3个：只有IR
                has_xrd = False
                has_sem = False
                ir_only_count += 1
                print(f"    -> 只有IR数据")
            elif mod_idx < 6:                  # 接下来3个：XRD + IR
                has_sem = False
                xrd_ir_only_count += 1
                print(f"    -> XRD + IR 数据")
            elif mod_idx < 9:                  # 再3个：IR + SEM
                has_xrd = False
                ir_sem_only_count += 1
                print(f"    -> IR + SEM 数据")
            else:                              # 其余：三种都有
                complete_count += 1
                print(f"    -> XRD + IR + SEM 数据")

            # 生成XRD
            if has_xrd:
                generate_xrd_csv(os.path.join(config.XRD_DIR, f"{sample_name}_xrd.csv"), cls)

            # 生成IR
            if has_ir:
                generate_ir_csv(os.path.join(config.IR_DIR, f"{sample_name}_ir.csv"), cls)

            # 生成SEM（表面和截面各1-2张）
            if has_sem:
                # 表面：1张图
                generate_sem_image(os.path.join(config.SEM_DIR, f"{sample_name}_surface_5000x.jpg"), cls)
                # 截面：1张图
                generate_sem_image(os.path.join(config.SEM_DIR, f"{sample_name}_cross_10000x.jpg"), cls)

                # 部分样品每种加第二张（不同放大倍数）
                if sample_count % 3 == 0:
                    generate_sem_image(os.path.join(config.SEM_DIR, f"{sample_name}_surface_2000x.jpg"), cls)
                    generate_sem_image(os.path.join(config.SEM_DIR, f"{sample_name}_cross_5000x.jpg"), cls)

            labels.append({"sample_name": sample_name, "label": cls})
            sample_count += 1

    # 保存标签文件
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(config.LABEL_FILE, index=False, encoding="utf-8-sig")

    print(f"\n模拟数据生成完成!")
    print(f"  总样品数: {sample_count}")
    print(f"  数据组合:")
    print(f"    三种都有(XRD+IR+SEM): {complete_count}个")
    print(f"    只有IR: {ir_only_count}个")
    print(f"    XRD+IR: {xrd_ir_only_count}个")
    print(f"    IR+SEM: {ir_sem_only_count}个")
    print(f"  数据位置: {config.DATA_DIR}")
    print(f"  标签文件: {config.LABEL_FILE}")
    print(f"\n现在可以运行:")
    print(f"  1. python step1_提取特征.py")
    print(f"  2. python step2_训练模型.py")
    print(f"  3. python step3_预测新样品.py --xrd data/xrd/GH001_xrd.csv --ir data/ir/GH001_ir.csv --sem_surface data/sem/GH001_surface_5000x.jpg --sem_cross data/sem/GH001_cross_10000x.jpg")
    print(f"  4. python step4_看看结果.py")
