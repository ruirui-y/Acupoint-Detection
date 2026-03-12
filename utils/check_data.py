import os


def check_yolo_data(img_dir, lbl_dir):
    images = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    labels = [os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')]

    print(f"--- 正在检测目录: {os.path.dirname(img_dir)} ---")
    print(f"找到图片: {len(images)} 张")
    print(f"找到标签: {len(labels)} 个")

    # 找没有标签的图片
    missing_labels = set(images) - set(labels)
    if missing_labels:
        print(f"❌ 警告：有 {len(missing_labels)} 张图片缺少标签文件！")
        print(f"缺失示例: {list(missing_labels)[:3]}")
    else:
        print("✅ 所有图片均有对应的标签文件。")

    # 检查标签内容格式
    if labels:
        sample_path = os.path.join(lbl_dir, labels[0] + '.txt')
        with open(sample_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"❌ 错误：样本标签 {labels[0]}.txt 是空的！")
            else:
                first_val = lines[0].split()[0]
                print(f"✅ 标签格式初步检查通过（样本首行类别 ID: {first_val}）")


# 使用你电脑上的绝对路径，注意 Windows 路径建议在引号前加 r 避免转义字符错误
check_yolo_data(r'H:\YJJ\PythonProject\multispectral-yolov8\data\my_68\rgb\images',
                r'H:\YJJ\PythonProject\multispectral-yolov8\data\my_68\rgb\labels')

check_yolo_data(r'H:\YJJ\PythonProject\multispectral-yolov8\data\my_68\depth\images',
                r'H:\YJJ\PythonProject\multispectral-yolov8\data\my_68\depth\labels')