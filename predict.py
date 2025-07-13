import os
from pathlib import Path
from ultralytics import YOLO


def predict_and_save_images(model, source_dir, output_dir):
    """
    对源目录中的所有图片进行预测并保存结果

    参数:
        model: 加载的YOLO模型
        source_dir: 包含测试图片的目录
        output_dir: 保存预测结果的目录
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [
        f for f in os.listdir(source_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"在 {source_dir} 中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片进行预测...")

    # 对每张图片进行预测并保存结果
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(source_dir, img_file)
        print(f"处理图片 {i}/{len(image_files)}: {img_file}")

        # 进行预测
        results = model.predict(img_path, conf=0.25)  # conf为置信度阈值

        # 保存预测结果
        for r in results:
            # 构建输出路径
            output_path = os.path.join(output_dir, f"pred_{img_file}")
            r.save(filename=output_path)
            print(f"预测结果已保存到: {output_path}")


if __name__ == '__main__':
    # 设置路径
    test_images_dir = "dataset/images/test"  # 测试图片目录
    output_dir = "runs/detect/predict"  # 预测结果输出目录

    # 加载模型（使用预训练的YOLOv8模型）
    # 可用模型: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO("runs/detect/train/weights/best.pt")

    print("开始预测...")
    predict_and_save_images(model, test_images_dir, output_dir)
    print("所有图片预测完成！")
