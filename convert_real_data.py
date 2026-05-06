# ============================================================
# convert_real_data.py —— 把 D:\date 的真实数据整理进项目目录
# ============================================================
# 做什么：
#   1. XRD: GBK编码 → UTF-8，重命名为 "样品名_xrd.csv"
#   2. IR:  科学计数法 → 普通浮点数，加表头，重命名为 "样品名_ir.csv"
#   3. SEM: 从子文件夹展平，重命名为 "样品名_surface_xxx.jpg/tif" 或 "样品名_cross_xxx.jpg/tif"
#   4. 生成 sample_labels.csv 模板（待你手动填写标签）
#
# 运行: python convert_real_data.py
# ============================================================

import os                                     # 文件操作
import shutil                                 # 文件复制
import re                                     # 正则表达式
import config                                 # 项目配置

# ============================================================
# 源数据和目标路径
# ============================================================
SRC_DIR = r"D:\date"                          # 真实数据总目录
SRC_XRD = os.path.join(SRC_DIR, "xrd")
SRC_IR = os.path.join(SRC_DIR, "ir")
SRC_SEM = os.path.join(SRC_DIR, "SEM")

# 清理旧模拟数据（只清数据，不清 config/models/scripts）
print("=" * 60)
print("真实数据整理工具")
print("=" * 60)

# 清空目标目录中的旧文件
for d in [config.XRD_DIR, config.IR_DIR, config.SEM_DIR]:
    if os.path.isdir(d):
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                os.remove(fp)

# 确保目标目录存在
os.makedirs(config.XRD_DIR, exist_ok=True)
os.makedirs(config.IR_DIR, exist_ok=True)
os.makedirs(config.SEM_DIR, exist_ok=True)


def auto_decode_file(filepath):
    """
    自动检测文件编码（UTF-8-BOM / GBK），返回文件行列表
    XRD仪器导出的CSV可能是GBK或UTF-8带BOM
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    # 先尝试 UTF-8 with BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw.decode("utf-8-sig")
            return text.splitlines(keepends=True)
        except UnicodeDecodeError:
            pass

    # 再尝试 GBK
    try:
        text = raw.decode("gbk")
        return text.splitlines(keepends=True)
    except UnicodeDecodeError:
        pass

    # 最后尝试 UTF-8
    try:
        text = raw.decode("utf-8")
        return text.splitlines(keepends=True)
    except UnicodeDecodeError:
        pass

    # 都失败就用 latin-1（不会报错但可能乱码）
    return raw.decode("latin-1").splitlines(keepends=True)


# ============================================================
# 1. 转换 XRD（自动检测编码 → UTF-8）
# ============================================================
print("\n--- 转换 XRD 数据 ---")
xrd_count = 0
for fname in sorted(os.listdir(SRC_XRD)):
    if fname.lower().endswith(".csv"):
        src_path = os.path.join(SRC_XRD, fname)
        # 自动检测编码
        lines = auto_decode_file(src_path)
        dst_path = os.path.join(config.XRD_DIR, fname)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        xrd_count += 1
        print(f"  已转换: {fname}")

print(f"  XRD 转换完成，共 {xrd_count} 个文件")


# ============================================================
# 2. 转换 IR（科学计数法 → 普通浮点，加表头）
# ============================================================
print("\n--- 转换 IR 数据 ---")
for fname in os.listdir(SRC_IR):
    if fname.lower().endswith(".csv"):
        src_path = os.path.join(SRC_IR, fname)
        dst_path = os.path.join(config.IR_DIR, fname)

        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with open(dst_path, "w", encoding="utf-8") as f:
            # 写入表头
            f.write("波数(cm-1),吸光度\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    # 科学计数法 → 浮点数
                    wavenum = float(parts[0])
                    absorb = float(parts[1])
                    f.write(f"{wavenum:.2f},{absorb:.6f}\n")
        print(f"  已转换: {fname}")

print(f"  IR 转换完成，共 {len(os.listdir(config.IR_DIR))} 个文件")


# ============================================================
# 3. 展平 SEM 文件
# ============================================================
print("\n--- 展平 SEM 文件 ---")
SEM_CATEGORIES = {
    "cross": "cross",            # cross文件夹 → cross
    "sarface": "surface",        # sarface文件夹 → surface（拼写错误，但保持原样）
}

sem_count = 0
for folder_name, sem_type in SEM_CATEGORIES.items():
    folder_path = os.path.join(SRC_SEM, folder_name)
    if not os.path.isdir(folder_path):
        print(f"  [跳过] 文件夹不存在: {folder_name}")
        continue

    for sample_dir in sorted(os.listdir(folder_path)):
        sample_dir_path = os.path.join(folder_path, sample_dir)
        if not os.path.isdir(sample_dir_path):
            continue

        # 从文件夹名提取样品名: "919-001-cross" → "919_001", "000-cross" → "000"
        # 也支持下划线分隔: "000_surface" → "000"
        prefix = sample_dir
        for sep in ["-", "_"]:
            for kw in [folder_name, SEM_CATEGORIES[folder_name]]:
                suffix = f"{sep}{kw}"
                if prefix.endswith(suffix):
                    prefix = prefix[:len(prefix) - len(suffix)]
                    break
            else:
                continue
            break
        # 统一用下划线: 919-001 → 919_001
        sample_name = prefix.replace("-", "_")

        # 复制所有图片文件
        for img_name in sorted(os.listdir(sample_dir_path)):
            img_src = os.path.join(sample_dir_path, img_name)
            if not os.path.isfile(img_src):
                continue

            # 只复制图片文件
            if not img_name.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")):
                continue

            # 去掉 "Image(" 前缀，提取放大倍数和序号
            # 处理格式: "Image(x1.2k)-5.jpg" 或 "000_Image(x30)_1.jpg" 或 "Image(x300)5.jpg"
            base = img_name
            if base.startswith("Image("):
                base = base[6:]  # 去掉 "Image("
            elif "_" in base and "Image(" in base:
                # 如 "000_Image(x30)_1.jpg" → 去掉样品名前缀
                base = base.split("Image(", 1)[1]

            # 提取放大倍数和序号: "x1.2k)-5" 或 "x30)-1" 或 "x300)5"
            import re as _re
            m = _re.match(r"(x[^)]*)\)\s*[-_]?\s*(\d+)", base)
            if m:
                mag = m.group(1).replace("-", "_")  # 放大倍数，如 x1.2k, x300
                num = m.group(2)                     # 序号，如 5, 1
                clean_name = f"{mag}_{num}"
            else:
                # fallback: 简单清理
                clean_name = base.replace(")", "_").replace("-", "_").rstrip("_")

            # 获取文件扩展名
            ext = os.path.splitext(img_name)[1]
            dst_name = f"{sample_name}_{sem_type}_{clean_name}{ext}"
            dst_path = os.path.join(config.SEM_DIR, dst_name)

            shutil.copy2(img_src, dst_path)
            sem_count += 1
            print(f"  已复制: {sample_dir}/{img_name} → {dst_name}")

print(f"  SEM 展平完成，共 {sem_count} 张图片")


# ============================================================
# 4. 生成标签模板
# ============================================================
print("\n--- 生成标签模板 ---")
import glob
import re

# 从 XRD 和 IR 文件提取所有样品名
xrd_samples = set()
ir_samples = set()
sem_samples = set()

for f in os.listdir(config.XRD_DIR):
    if f.lower().endswith("_xrd.csv"):
        xrd_samples.add(f.replace("_xrd.csv", "").replace("_XRD.csv", ""))

for f in os.listdir(config.IR_DIR):
    if f.lower().endswith("_ir.csv"):
        # 去掉 _ir.CSV 或 _ir.csv 后缀（7个字符）
        ir_samples.add(f[:-7])

for f in os.listdir(config.SEM_DIR):
    if f.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")):
        # 从文件名提取样品名: "919_001_surface_x1.2k_5.jpg" → "919_001"
        # 匹配第一个 _surface 或 _cross 之前的部分
        match = re.match(r"^(.+?)_(?:surface|cross)", f, re.IGNORECASE)
        if match:
            sem_samples.add(match.group(1))

all_samples = xrd_samples | ir_samples | sem_samples

# 生成标签CSV（按样品名排序）
label_path = config.LABEL_FILE
with open(label_path, "w", encoding="utf-8-sig") as f:
    f.write("sample_name,label\n")
    for sample in sorted(all_samples):
        data_types = []
        if sample in xrd_samples:
            data_types.append("XRD")
        if sample in ir_samples:
            data_types.append("IR")
        if sample in sem_samples:
            data_types.append("SEM")
        f.write(f"{sample},  # 有{'+'.join(data_types)}数据，请填入老化级别\n")

print(f"  标签模板已生成: {label_path}")
print(f"  共 {len(all_samples)} 个样品:")
for sample in sorted(all_samples):
    data_types = []
    if sample in xrd_samples:
        data_types.append("XRD")
    if sample in ir_samples:
        data_types.append("IR")
    if sample in sem_samples:
        data_types.append("SEM")
    print(f"    {sample}: {', '.join(data_types)}")


# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 60)
print("数据整理完成!")
print("=" * 60)
print("\n下一步:")
print("  1. 打开 data/sample_labels.csv，手动填入每个样品的老化级别")
print("  2. 运行: python step1_提取特征.py")
print("  3. 运行: python step2_训练模型.py")
