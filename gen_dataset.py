import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path

# 配置参数
DATASET_DIR = "dataset"  # 数据集根目录
IMAGE_SIZE = (128, 128)  # 图像尺寸
NUM_IMAGES = 50000  # 总图像数量
TRAIN_RATIO = 0.7  # 训练集比例
VAL_RATIO = 0.2  # 验证集比例
TEST_RATIO = 0.1  # 测试集比例

# 字符集：数字 + 大写字母 + 小写字母
CHARACTERS = [str(i) for i in range(10)] + \
             [chr(ord('A') + i) for i in range(26) if i != 14] + \
             [chr(ord('a') + i) for i in range(26) if i != 14]


# 创建目录结构
def create_directories():
    # 创建图像目录
    for folder in ["images/train", "images/val", "images/test"]:
        Path(f"{DATASET_DIR}/{folder}").mkdir(parents=True, exist_ok=True)

    # 创建标签目录
    for folder in ["labels/train", "labels/val", "labels/test"]:
        Path(f"{DATASET_DIR}/{folder}").mkdir(parents=True, exist_ok=True)


# 获取字体并调整字符大小
def get_and_adjust_font(char):
    # 尝试加载系统字体
    fonts = [
        "arial.ttf", "arialbd.ttf", "ariali.ttf",
        "times.ttf", "timesbd.ttf", "timesi.ttf",
        "cour.ttf", "courbd.ttf", "couri.ttf",
        "verdana.ttf", "verdanab.ttf", "verdanai.ttf"
    ]

    # 初始字体大小
    font_size = 40
    best_font = None
    best_char_img = None
    img_size = (random.randint(24, 48), random.randint(24, 48))
    for _ in range(5):  # 最多尝试5次调整大小
        try:
            font_path = random.choice(fonts)
            font = ImageFont.truetype(font_path, font_size)

            # 1. 创建临时图像来测量字符大小
            temp_img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)

            # 2. 获取字符边界框
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]

            # 检查字符大小是否接近目标大小
            if char_width > 0 and char_height > 0:
                # 3. 创建字符图像
                char_img = Image.new('RGBA', (char_width, char_height), (0, 0, 0, 0))
                char_draw = ImageDraw.Draw(char_img)
                # 4. 截取字符图像（考虑边界框偏移）
                char_draw.text((-bbox[0], -bbox[1]), char, font=font, fill=(0, 0, 0, 255))

                # 5. 强制调整为32x32大小
                char_img = char_img.resize(img_size, Image.Resampling.LANCZOS)
                return char_img, img_size

                # 如果字符大小不合适，调整字体大小
                scale_factor = min(img_size[0] / char_width, img_size[1] / char_height)
                font_size = int(font_size * scale_factor * 0.9)
                font_size = max(10, min(100, font_size))
        except:
            continue

    # 如果调整失败，使用默认字体
    try:
        font = ImageFont.load_default().font_variant(size=32)
        char_img = Image.new('RGBA', img_size, (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((0, 0), char, font=font, fill=(0, 0, 0, 255))
        return char_img, img_size
    except:
        # 创建简单字符图像作为回退
        char_img = Image.new('RGBA', img_size, (0, 0, 0, 255))
        draw = ImageDraw.Draw(char_img)
        draw.text((8, 8), char, fill=(255, 255, 255, 255))
        return char_img, img_size


# 生成单张图像和标签
def generate_image_and_label(img_id):
    # 创建空白图像
    image = Image.new('RGB', IMAGE_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 随机选择字符
    char = random.choice(CHARACTERS)
    char_idx = CHARACTERS.index(char)

    # 获取调整后的字符图像
    char_img, image_size = get_and_adjust_font(char)

    # 固定位置
    position = (random.randint(0, 80), random.randint(0, 80))

    # 将字符图像粘贴到主图像
    image.paste(char_img, position, char_img)

    # 固定边界框（字符大小32x32）
    abs_x_min = position[0]
    abs_y_min = position[1]
    abs_x_max = position[0] + image_size[0]
    abs_y_max = position[1] + image_size[1]

    # 计算归一化坐标 (YOLO格式)
    center_x = ((abs_x_min + abs_x_max) / 2) / IMAGE_SIZE[0]
    center_y = ((abs_y_min + abs_y_max) / 2) / IMAGE_SIZE[1]
    width = image_size[0] / IMAGE_SIZE[0]
    height = image_size[1] / IMAGE_SIZE[1]

    # 添加干扰线
    for _ in range(random.randint(3, 8)):
        start = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        end = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.line([start, end], fill=line_color, width=random.randint(1, 3))

    # 添加随机噪点
    if random.random() > 0.7:
        for _ in range(100):
            px = (random.randint(0, IMAGE_SIZE[0] - 1), random.randint(0, IMAGE_SIZE[1] - 1))
            draw.point(px, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # YOLO标签格式: <class_id> <center_x> <center_y> <width> <height>
    yolo_label = f"{char_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

    return image, char_idx, yolo_label


# 生成数据集
def generate_dataset():
    create_directories()

    # 划分数据集
    indices = list(range(NUM_IMAGES))
    random.shuffle(indices)

    train_end = int(NUM_IMAGES * TRAIN_RATIO)
    val_end = train_end + int(NUM_IMAGES * VAL_RATIO)

    # 类别分布统计
    class_counts = {i: 0 for i in range(len(CHARACTERS))}

    for idx in range(NUM_IMAGES):
        image, char_idx, yolo_label = generate_image_and_label(idx)

        # 更新类别计数
        class_counts[char_idx] += 1

        # 确定数据集划分
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"

        # 保存图像
        image_filename = f"{char_idx}_{idx:05d}.jpg"
        image.save(f"{DATASET_DIR}/images/{split}/{image_filename}")

        # 保存YOLO标签
        label_filename = f"{char_idx}_{idx:05d}.txt"
        with open(f"{DATASET_DIR}/labels/{split}/{label_filename}", "w") as f:
            f.write(yolo_label)

        if (idx + 1) % 100 == 0:
            print(f"Generated {idx + 1}/{NUM_IMAGES} images and labels")

    # 打印类别分布
    print("\nClass distribution:")
    for char, idx in zip(CHARACTERS, range(len(CHARACTERS))):
        print(f"{char}: {class_counts[idx]} samples")

    return class_counts


# 创建YOLO数据集配置文件
def create_yaml_config(class_counts):
    # 创建类别名称列表
    names_list = "[" + ", ".join([f"'{char}'" for char in CHARACTERS]) + "]"

    content = f"""# YOLO dataset configuration
path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(CHARACTERS)}

# Class names
names: {names_list}

# Class distribution
class_distribution:
"""
    for char, idx in zip(CHARACTERS, range(len(CHARACTERS))):
        content += f"  '{char}': {class_counts[idx]}\n"

    with open(f"{DATASET_DIR}/dataset.yaml", "w") as f:
        f.write(content)


if __name__ == "__main__":
    class_counts = generate_dataset()
    create_yaml_config(class_counts)
    print(f"\nDataset generated at {os.path.abspath(DATASET_DIR)}")
    print(f"YOLO config file created at {os.path.abspath(DATASET_DIR)}/dataset.yaml")
