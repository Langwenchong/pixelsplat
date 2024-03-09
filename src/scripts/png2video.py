import os
import cv2

def images_to_video(input_folder, output_video_path, fps=24):
    # 获取文件夹中所有PNG图片的路径
    image_paths = [os.path.join(input_folder, 'color', f) for f in os.listdir(os.path.join(input_folder, 'color')) if f.endswith('.png')]
    image_paths.sort()  # 确保图片按顺序合成视频

    # 读取第一张图片，获取图片尺寸
    img = cv2.imread(image_paths[0])
    height, width, _ = img.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历图片，将每一帧写入视频
    for image_path in image_paths:
        img = cv2.imread(image_path)
        out.write(img)

    # 释放资源
    out.release()
    cv2.destroyAllWindows()

# 指定输入文件夹路径
input_root_folder = '/data/lwc/PythonProjects/pixelsplat/outputs/test/re10k'

# 遍历每一个文件夹
for folder_name in os.listdir(input_root_folder):
    folder_path = os.path.join(input_root_folder, folder_name)
    if os.path.isdir(folder_path):
        # 生成视频的输出路径
        output_video_path = os.path.join(folder_path, 'video.mp4')
        
        # 调用函数生成视频
        images_to_video(folder_path, output_video_path)
