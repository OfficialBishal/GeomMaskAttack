import os
import time
import math
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
import utils
import deepfool
import line
import spot
import fastspot


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Layer 1
        self.cnn11 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.cnn12 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)

        # Layer 2
        self.cnn21 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=0)
        self.cnn22 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(32)

        # Layer 3
        self.cnn31 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=0)
        self.cnn32 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(64)

        # Layer 4
        self.cnn41 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=0)
        self.cnn42 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=0)
        self.cnn43 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=0)
        self.batchnorm4 = nn.BatchNorm2d(128)

        # Flatten
        self.fc1 = nn.Linear(in_features=2304, out_features=500)
        self.droput = nn.Dropout(p=0.5)  # Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=500, out_features=13)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)

    def forward(self, x):
        # Layer 1
        out = self.cnn11(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.cnn12(out)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Layer 2
        out = self.cnn21(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.cnn22(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Layer 3
        out = self.cnn31(out)
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.cnn32(out)
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Layer 4
        out = self.cnn41(out)
        out = self.batchnorm4(out)
        out = self.relu(out)
        out = self.cnn42(out)
        out = self.batchnorm4(out)
        out = self.relu(out)
        out = self.cnn43(out)
        out = self.batchnorm4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Flattening is done here with .view() -> (batch_size, 128*6*3) = (100, 2304)
        # -1 will automatically update the batchsize as 100; 2304 flattens 128,6,3
        out = out.view(-1, 2304)
        # Then we forward through our fully connected layer
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        return out


def load_model(PATH_m):
    device = torch.device('cpu')

    # Loading the trained network
    # PATH_m = 'models/model-epoch-17.pth'
    model = torch.load(PATH_m, map_location=device)

    # Switch to evaluation mode
    model.eval()    # print(model.eval())

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    return model, loss_fn, optimizer


def load_data():
    # Loading dataset (Image size = 160x105)
    data_path = 'images/Segments_Sorted'
    batch_size = 1
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'F')

    transform_ori = transforms.Compose(
        [transforms.ToTensor()])  # convert the image to a Tensor
    dataset = datasets.ImageFolder(data_path, transform=transform_ori)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size)  # batch_size=len(dataset)

    return dataset_loader


def image_detail(model, im, labels):
    # show images
    print(f'labels: {utils.classes[labels]}')
    print(f'shape: {im.shape}')
    utils.imshow(torchvision.utils.make_grid(im))
    utils.imshow_transform(im, labels)
    # imshow(torchvision.utils.make_grid(images))     # imshow(images[0,:,:,:])

    # classify the image and extract the predictions
    print("[INFO] classifying image...")
    utils.classification_prediction(model, im.view(1, 3, 160, 105))


def mode_execution(mode, model, loss_fn, optimizer, im, labels):

    # print("Modes: 'deepfool', 'spot', 'vline', 'hline'")
    # mode = input("Enter mode: ")  # To get mode selection from input

    record = dict()
    if (mode == 'fgsm'):
        print('[INFO] FGSM:')

    elif (mode == 'deepfool'):
        print('[INFO] Deepfool:')
        since = time.time()

        r, loop_i, label_orig, label_pert, pert_image = deepfool.deepfool(
            im, model, num_classes=13, overshoot=0.02, max_iter=50)

        time_elapsed = time.time() - since
        print('Deepfool completed in {:.3f}m {:.10f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(
            f'Loop: {loop_i}\nEstimated Label: {label_orig}\nPerturbed Label: {label_pert}')
        record['shape_param'] = loop_i
        record['time'] = time_elapsed

    elif (mode == 'spot'):
        print('[INFO] Spot')
        print("Processing...")
        success = False
        radius = 1
        since = time.time()
        for radius_iter in range(1, int(105/2), 1):
            if (success == False):
                pert_image, label_pert, success, top_prob, mse = spot.main(
                    model, loss_fn, im, labels, radius_iter)
                radius = radius_iter
            elif (success == True):
                break
        if (success == False):
            radius = 0
        time_elapsed = time.time() - since
        print("Spot completed at radius '{}' in {:.3f}m {:.10f}s".format(
            radius, time_elapsed // 60, time_elapsed % 60))
        record['shape_param'] = radius
        record['time'] = time_elapsed
        record['top_prob'] = top_prob
        record['mse'] = mse

    elif (mode == 'fastspot'):
        print('[INFO] Fast Spot')
        print("Processing...")
        success = False
        since = time.time()
        for i in range(5):
            if (success == False):
                pert_image, label_pert, radius, success, top_prob, mse = fastspot.main(
                    model, loss_fn, im, labels)
            elif (success == True):
                print(f'{i} Success')
                break
        if (success == False):
            radius = 0
        time_elapsed = time.time() - since
        print("Spot completed at radius '{}' in {:.3f}m {:.10f}s".format(
            radius, time_elapsed // 60, time_elapsed % 60))
        record['shape_param'] = radius
        record['time'] = time_elapsed
        record['top_prob'] = top_prob
        record['mse'] = mse

    elif (mode == 'vline'):
        print('[INFO] Vertical Line')
        print("Processing...")
        vline = True
        hline = False
        success = False
        thickness = 1
        since = time.time()
        for thickness_iter in range(1, int(105/2), 1):
            if (success == False):
                pert_image, label_pert, success, top_prob, mse = line.main(
                    model, loss_fn, im, labels, vline, hline, thickness_iter)
                thickness = thickness_iter
            elif (success == True):
                break
        if (success == False):
            thickness = 0
        time_elapsed = time.time() - since
        print("Vertical Line completed at thickness '{}' in {:.3f}m {:.10f}s".format(
            thickness, time_elapsed // 60, time_elapsed % 60))
        record['shape_param'] = thickness
        record['time'] = time_elapsed
        record['top_prob'] = top_prob
        record['mse'] = mse

    elif (mode == 'hline'):
        print('[INFO] Horizontal Line')
        print("Processing...")
        vline = False
        hline = True
        success = False
        thickness = 1
        since = time.time()
        for thickness_iter in range(1, int(160/2), 1):
            if (success == False):
                pert_image, label_pert, success, top_prob, mse = line.main(
                    model, loss_fn, im, labels, vline, hline, thickness_iter)
                thickness = thickness_iter
            elif (success == True):
                break
        if (success == False):
            thickness = 0
        time_elapsed = time.time() - since
        print("Horizontal Line completed at thickness '{}' in {:.3f}m {:.10f}s".format(
            thickness, time_elapsed // 60, time_elapsed % 60))
        record['shape_param'] = thickness
        record['time'] = time_elapsed
        record['top_prob'] = top_prob
        record['mse'] = mse

    return pert_image, label_pert, record


def main():
    utils.clearConsole()
    print("WELCOME TO THE PROGRAM\n"+"-"*30)

    op_path = 'outputs'

    # Select mode
    mode = 'spot'

    if (mode == 'spot' or mode == 'vline'):
        temp_const = int(105/2)
    elif (mode == 'hline'):
        temp_const = int(160/2)

    # Loading model
    model_path = 'models/gif/'  # path to the model from which adversarial samples for gif is to be generated
    for model_name in os.listdir(model_path):
        record_shape_param_matrix1 = np.zeros((13, temp_const), dtype=int)
        record_shape_param_matrix2 = np.zeros((13, 13, temp_const), dtype=int)
        record_label_pert_matrix = np.zeros((13, 13), dtype=int)
        record_label_logit_matrix = np.zeros((13, 13), dtype=float)
        record_MSE_matrix = np.zeros((13, 13), dtype=float)
        record_time = []
        record_shape_param = []
        record_label_pert = []
        model, loss_fn, optimizer = load_model(model_path+model_name)

        # Loading data
        dataset_loader = load_data()
        for i, data in enumerate(dataset_loader):
            # if (i >= 0 and i <= 5) or (i >= 90 and i <= 95) or (i >= 223 and i <= 228) or (i >= 305 and i <= 310) or (i >= 362 and i <= 367) or (i >= 395 and i <= 400) or (i >= 453 and i <= 459) or (i >= 510 and i <= 515) or (i >= 575 and i <= 580) or (i >= 590 and i <= 595) or (i >= 849 and i <= 856) or (i >= 950 and i <= 955) or (i >= 1026 and i <= 1032):
            if (i == 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels = data
                # [1, 3, 160, 105] -> [3, 160, 105]
                im = images.view(3, 160, 105)

                # Input Image Details
                # print('[INFO] Input Image:')
                # image_detail(model, im, labels)
                # print('-'*100)

                # Adversarial Algorithm
                pert_image, label_pert, record = mode_execution(
                    mode, model, loss_fn, optimizer, im, labels)
                print('-'*100)

                # Recording outputs
                record_shape_param_matrix1[labels.item(
                )][record['shape_param']] += 1
                record_shape_param_matrix2[labels.item(
                )][label_pert][record['shape_param']] += 1
                record_label_pert_matrix[labels.item()][label_pert] += 1
                record_label_logit_matrix[labels.item(
                )][label_pert] += record['top_prob'].values.item()
                record_MSE_matrix[labels.item()][label_pert] += record['mse']

                record_time.append(record['time'])
                record_shape_param.append(record['shape_param'])
                record_label_pert.append(label_pert)

                # Perturbed Image Details
                # print('[INFO] Perturbed Image:')
                # [1, 3, 160, 105] -> [3, 160, 105]
                pert_image = pert_image.view(3, 160, 105)
                # image_detail(model, pert_image, label_pert)

                # Saving the perturbed image
                output_path = f'{op_path}/gif3/'
                output_file_name = f"{mode}-{i}-O({utils.classes[label_pert]})-t({int(record['shape_param'])}).png"
                os.makedirs(output_path, exist_ok=True)
                save_image(pert_image, output_path + output_file_name)


if __name__ == '__main__':
    main()


# Copy and paste this in sopt.py or line.py
# pert_img = create_spot(img.copy(), center_i, radius, rgb)
# pert_image = numpy_PIL_tensor(pert_img)
#     # Saving the perturbed image
# output_path = 'outputs/gif3/'
# output_file_name = f"{radius}-{center_i}.png"
# os.makedirs(output_path, exist_ok=True)
# pert_image = pert_image.view(3, 160, 105)
# save_image(pert_image, output_path + output_file_name)