#!/usr/env python
# -*- coding:utf-8 -*-
# author:qianqian time:4/19/2024

import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=30):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_frame_count = 0

    while frame_count < total_frames:
        # 读取下一帧
        success, frame = cap.read()

        # 如果读取失败，跳出循环
        if not success:
            break

        # 每隔 frame_interval 帧保存一次
        if frame_count % frame_interval == 0:
            # 保存帧为图片
            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1

        frame_count += 1

    # 释放视频捕获对象
    cap.release()

# 使用示例
video_path = 'D:/OneDrive - University of Central Florida/RTOR/2.REF_Other characteristics/1.Data/0.Intersection and T-intersection/Video/publix050422PM01.mp4'  # 替换为你的视频文件路径
output_folder = 'D:/01_30fps'  # 替换为你想要保存帧的文件夹路径
extract_frames(video_path, output_folder)
