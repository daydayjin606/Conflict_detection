#!/usr/env python
# -*- coding:utf-8 -*-
# author:qianqian time:7/5/2024

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import random
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torchvision.models import ResNet101_Weights


class CustomDataset(Dataset):
    def __init__(self, folder0_paths, folder1_paths, transform=None, ratio=0.3):
        self.folder0_paths = folder0_paths if isinstance(folder0_paths, list) else [folder0_paths]
        self.folder1_paths = folder1_paths if isinstance(folder1_paths, list) else [folder1_paths]
        self.transform = transform
        self.ratio = ratio

        # List all files in Folder 0 paths
        self.folder0_files = []
        for path in self.folder0_paths:
            self.folder0_files.extend([os.path.join(path, file) for file in os.listdir(path)])

        # List all files in Folder 1 paths
        self.folder1_files = []
        for path in self.folder1_paths:
            self.folder1_files.extend([os.path.join(path, file) for file in os.listdir(path)])

        # Calculate the number of samples to draw from each folder
        print("Number of samples in file0:", len(self.folder0_files))
        print("Number of samples in file1:", len(self.folder1_files))
        sample_rate = len(self.folder0_files) / len(self.folder1_files)
        print('Real sample rate: file0/file1:', sample_rate)

        if sample_rate >= 1:
            if self.ratio >= 1:
                # ratio < 1, more samples from folder1
                num_samples_from_folder1 = len(self.folder1_files)
                num_samples_from_folder0 = int(num_samples_from_folder1 * self.ratio)
            else:

                # ratio means sample sample file 0:1, so ratio >=1, representing more sample from folder0
                num_samples_from_folder0 = len(self.folder0_files)
                num_samples_from_folder1 = int(num_samples_from_folder0 / self.ratio)
        else:
            if self.ratio >= 1:
                num_samples_from_folder0 = len(self.folder0_files)
                num_samples_from_folder1 = int(num_samples_from_folder0 / self.ratio)

            else:
                num_samples_from_folder1 = len(self.folder1_files)
                num_samples_from_folder0 = int(num_samples_from_folder1 * self.ratio)

        # Randomly sample files from each folder
        self.sampled_folder0_files = random.sample(self.folder0_files, num_samples_from_folder0)
        self.sampled_folder1_files = random.sample(self.folder1_files, num_samples_from_folder1)

        # Combine the sampled files
        self.sampled_files = self.sampled_folder0_files + self.sampled_folder1_files
        random.shuffle(self.sampled_files)  # Shuffle the combined list

        # Print the number of samples used from each folder
        print(f"\nNumber of samples from folder0: {num_samples_from_folder0}")
        print(f"Number of samples from folder1: {num_samples_from_folder1}")

    def __len__(self):
        # Return total number of samples
        return len(self.sampled_files)

    def __getitem__(self, idx):
        # Load tensor and label based on sampled_files
        file_path = self.sampled_files[idx]
        tensor = torch.load(file_path)
        label = 0 if file_path in self.sampled_folder0_files else 1

        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


class CustomTransform:
    def __init__(self, crop_size=None, flip_prob=0.5):
        self.crop_size = crop_size
        self.flip_prob = flip_prob

    def __call__(self, tensor):
        # Random horizontal flip
        if torch.rand(1).item() < self.flip_prob:
            tensor = tensor.flip(-1)

        # Random crop
        if self.crop_size:
            _, h, w = tensor.shape
            new_h, new_w = self.crop_size
            if h >= new_h and w >= new_w:
                top = torch.randint(0, h - new_h + 1, (1,)).item()
                left = torch.randint(0, w - new_w + 1, (1,)).item()
                tensor = tensor[:, top:top + new_h, left:left + new_w]
            else:
                # If image is smaller than crop size, skip cropping
                pass

        return tensor


def main():
    custom_transform = CustomTransform(crop_size=(256, 256), flip_prob=0.5)

    folder0_path = [r"D:\Video_frame\video_frames_a\MA\3fps\0",
                    r"D:\Video_frame\video_frames_b\MA\3fps\0",
                    r"D:\Video_frame\video_frames_c\MA\3fps\0",
                    r"D:\Video_frame\video_frames_d\MA\3fps\0"]

    folder1_path = [r"D:\Video_frame\video_frames_a\MA\3fps\1",
                    r"D:\Video_frame\video_frames_b\MA\3fps\1",
                    r"D:\Video_frame\video_frames_c\MA\3fps\1",
                    r"D:\Video_frame\video_frames_d\MA\3fps\1"]

    full_dataset = CustomDataset(folder0_paths=folder0_path,
                                 folder1_paths=folder1_path,
                                 transform=custom_transform,
                                 ratio=3)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers= 8)
    print('Train data loaded!')

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers = 8)
    print('Test data loaded!')

    # model = models.resnet101(pretrained=True)
    model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

    # # frozen pretained layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # replace fully connected layer
    num_features = model.fc.in_features
    num_classes = 2
    model.fc = nn.Linear(num_features, num_classes)

    # 修改为6通道，第一层
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride = (2, 2), padding= (3, 3), bias=False)

    # 将原始权重复制到新的卷积层中
    with torch.no_grad():
        model.conv1.weight[:, :3, :, :] = original_conv1.weight
        model.conv1.weight[:, 3:, :, :] = original_conv1.weight

    model = model.cuda()
    model = nn.DataParallel(model)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma= 0.1)

    # Train
    num_epochs = 100
    train_losses = []

    print('Starting training!')
    with open(r'D:\OneDrive - University of Central Florida\RTOR\3.CNN\MA_ratio3_3fps_RESNET_cropped.txt', 'w') as f:
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch + 1}')
            f.write(f'Epoch: {epoch + 1}\n')

            model.train()
            train_running_loss = 0.0
            train_predictions = []
            train_labels = []

            # for inputs, labels in tqdm(train_loader):
            for i, (images, labels) in enumerate(tqdm(train_loader)):

                labels = labels.long().cuda()
                images = images.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_running_loss += loss.item()

                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                train_predictions.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            scheduler.step()

            train_loss = train_running_loss / len(train_loader)
            train_losses.append(train_loss)

            cm_train = confusion_matrix(train_labels, train_predictions)
            print("Confusion Matrix:")
            print(cm_train)
            print('Train Performance:')
            f.write("Confusion Matrix:\n")
            f.write(f"{cm_train}\n")
            f.write('Train Performance:\n')
            train_metrics = calculate_classification_metrics(train_labels, train_predictions)
            f.write(f"{json.dumps(train_metrics)}\n")
            f.write(f'\nCurrent epoch Train loss: {train_running_loss}\n\n')
            print('\n')

            print(f'Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]}')

            print(f'Epoch {epoch + 1 }/{num_epochs} - Training loss: {train_loss}')
            print('\n')

    torch.save(model.state_dict(), 'MA_ratio3_3fps_RESNET_cropped.pth')
    print('Model saved!')

    print('Starting final testing!')
    evaluate_model(model, test_loader, criterion)

    visualize_loss(train_losses)


def visualize_loss(train_losses):
    plt.plot(train_losses, label='Training loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.show()


def calculate_classification_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_predictions, test_labels = [], []
    test_running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            labels = labels.long().cuda()
            images = images.cuda()

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        test_loss = test_running_loss / len(test_loader)

    cm_test = confusion_matrix(test_labels, test_predictions)
    print("\nTest Confusion Matrix:")
    print(cm_test)
    print('Test Performance:')
    test_metrics = calculate_classification_metrics(test_labels, test_predictions)
    print(f"Test loss: {test_loss:.4f}")
    print(json.dumps(test_metrics, indent=4))


if __name__ == "__main__":
    start_time = time.time()

    main()

    print(f"Execution time: {time.time() - start_time:.2f} seconds")
