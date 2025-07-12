import os
import random
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path

# 配置参数
DATASET_DIR = "dataset"  # 数据集根目录
IMAGE_SIZE = (128, 128)  # 图像尺寸
NUM_IMAGES = 5000  # 总图像数量
TRAIN_RATIO = 0.7  # 训练集比例
VAL_RATIO = 0.2  # 验证集比例
TEST_RATIO = 0.1  # 测试集比例

# 字符集：数字 + 大写字母 + 小写字母
CHARACTERS = [str(i) for i in range(10)] + \
             [chr(ord('A') + i) for i in range(26)] + \
             [chr(ord('a') + i) for i in range(26)]


# 创建目录结构
def create_directories():
    for folder in ["images/train", "images/val", "images/test"]:
        Path(f"{DATASET_DIR}/{folder}").mkdir(parents=True, exist_ok=True)


# 获取随机字体
def get_random_font():
    try:
        # 尝试加载系统字体
        fonts = [
            "arial.ttf", "arialbd.ttf", "ariali.ttf",
            "times.ttf", "timesbd.ttf", "timesi.ttf",
            "cour.ttf", "courbd.ttf", "couri.ttf",
            "verdana.ttf", "verdanab.ttf", "verdanai.ttf"
        ]
        font_path = random.choice(fonts)
        return ImageFont.truetype(font_path, random.randint(100, 180))
    except:
        # 回退到默认字体
        return ImageFont.load_default()


# 生成单张图像和标签
def generate_image_and_label(img_id):
    # 创建空白图像
    image = Image.new('RGB', IMAGE_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 随机选择字符
    char = random.choice(CHARACTERS)
    char_idx = CHARACTERS.index(char)

    # 随机字体和颜色
    font = get_random_font()
    text_color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

    # 创建字符图层
    bbox = font.getbbox(char)
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    # 创建透明图层用于字符变换
    char_layer = Image.new('RGBA', (char_width, char_height), (0, 0, 0, 0))
    char_draw = ImageDraw.Draw(char_layer)
    char_draw.text((-bbox[0], -bbox[1]), char, fill=text_color, font=font)

    # 随机缩放 (1-3倍)
    scale_factor = random.uniform(1.0, 3.0)
    scaled_width = int(char_width * scale_factor)
    scaled_height = int(char_height * scale_factor)
    scaled_char = char_layer.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    # 随机旋转 (0-360度)
    rotation_angle = random.randint(0, 360)
    rotated_char = scaled_char.rotate(rotation_angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    # 获取旋转后字符的实际边界框
    char_bbox = rotated_char.getbbox()
    if char_bbox is None:  # 如果没有有效像素，使用整个图像
        char_bbox = (0, 0, rotated_char.width, rotated_char.height)

    # 计算实际字符区域的尺寸
    actual_width = char_bbox[2] - char_bbox[0]
    actual_height = char_bbox[3] - char_bbox[1]

    # 确保字符在图像内 - 计算最大可用位置
    max_x = IMAGE_SIZE[0] - actual_width
    max_y = IMAGE_SIZE[1] - actual_height

    # 如果字符太大，缩小到合适尺寸
    if max_x < 0 or max_y < 0:
        scale_factor = min(IMAGE_SIZE[0] / actual_width, IMAGE_SIZE[1] / actual_height, 1.0)
        scaled_width = int(actual_width * scale_factor)
        scaled_height = int(actual_height * scale_factor)
        rotated_char = rotated_char.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        char_bbox = rotated_char.getbbox()
        if char_bbox is None:
            char_bbox = (0, 0, scaled_width, scaled_height)
        actual_width = char_bbox[2] - char_bbox[0]
        actual_height = char_bbox[3] - char_bbox[1]
        max_x = IMAGE_SIZE[0] - actual_width
        max_y = IMAGE_SIZE[1] - actual_height

    # 随机位置 (确保字符完全在图像内)
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0

    # 调整位置以匹配实际字符区域
    x -= char_bbox[0]
    y -= char_bbox[1]

    # 将字符粘贴到主图像
    image.paste(rotated_char, (x, y), rotated_char)

    # 计算实际边界框在图像中的位置
    abs_x_min = x + char_bbox[0]
    abs_y_min = y + char_bbox[1]
    abs_x_max = abs_x_min + actual_width
    abs_y_max = abs_y_min + actual_height

    # 确保边界框在图像范围内
    abs_x_min = max(0, min(IMAGE_SIZE[0] - 1, abs_x_min))
    abs_y_min = max(0, min(IMAGE_SIZE[1] - 1, abs_y_min))
    abs_x_max = max(0, min(IMAGE_SIZE[0] - 1, abs_x_max))
    abs_y_max = max(0, min(IMAGE_SIZE[1] - 1, abs_y_max))

    # 计算归一化坐标 (YOLO格式)
    center_x = ((abs_x_min + abs_x_max) / 2) / IMAGE_SIZE[0]
    center_y = ((abs_y_min + abs_y_max) / 2) / IMAGE_SIZE[1]
    width = (abs_x_max - abs_x_min) / IMAGE_SIZE[0]
    height = (abs_y_max - abs_y_min) / IMAGE_SIZE[1]

    # 确保坐标在[0,1]范围内
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

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

    # 添加随机椭圆作为额外干扰
    if random.random() > 0.5:
        x0 = random.randint(0, IMAGE_SIZE[0] // 2)
        y0 = random.randint(0, IMAGE_SIZE[1] // 2)
        x1 = random.randint(IMAGE_SIZE[0] // 2, IMAGE_SIZE[0])
        y1 = random.randint(IMAGE_SIZE[1] // 2, IMAGE_SIZE[1])
        ellipse_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse([x0, y0, x1, y1], outline=ellipse_color, width=random.randint(1, 2))

    return image, char_idx


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
        img_id = f"{idx:05d}"
        image, char_idx = generate_image_and_label(idx)

        # 更新类别计数
        class_counts[char_idx] += 1

        # 确定数据集划分
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"

        # 保存图像和标签
        image.save(f"{DATASET_DIR}/images/{split}/{char_idx}_{idx:05d}.jpg")

        if (idx + 1) % 100 == 0:
            print(f"Generated {idx + 1}/{NUM_IMAGES} images")

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