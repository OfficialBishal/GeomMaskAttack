import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils


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

# Returns logit and index of top 1 prediction.
def get_confidence(model, data):
    output = model(data)

    sm = torch.nn.Softmax()
    probabilities = torch.nn.Softmax(dim=-1)(output)
    top1_prob = torch.topk(probabilities, 1)

    return top1_prob


def main():
    utils.clearConsole()
    print("WELCOME TO THE PROGRAM\n"+"-"*30)

    path = 'outputs/Result/Adversarial Training/Original Dataset'

    # Loading model
    model_path = 'models/checkpoints/adv_train/'
    for model_name in os.listdir(model_path):
        record_label_matrix = np.zeros((13, 13), dtype=int)
        record_label_logit_matrix = np.zeros((13, 13), dtype=float)
        record_MSE_matrix = np.zeros((13, 13), dtype=float)
        record_label_pred = []
        model, loss_fn, optimizer = load_model(model_path+model_name)

        # Loading data
        dataset_loader = load_data()
        for i, data in enumerate(dataset_loader):
            # if (i >= 0 and i <= 5) or (i >= 90 and i <= 95) or (i >= 223 and i <= 228) or (i >= 305 and i <= 310) or (i >= 362 and i <= 367) or (i >= 395 and i <= 400) or (i >= 453 and i <= 459) or (i >= 510 and i <= 515) or (i >= 575 and i <= 580) or (i >= 590 and i <= 595) or (i >= 849 and i <= 856) or (i >= 950 and i <= 955) or (i >= 1026 and i <= 1032):
            if (True):
                # get the inputs; data is a list of [inputs, labels]
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                pred = predicted.item()
                
                top_prob = get_confidence(model, images)
                print(f'Prob: {top_prob.values.item()}, Index: {top_prob.indices.item()}')

                im = images.view(3, 160, 105)
                im = transforms.ToPILImage()(im).convert("RGB")  # tensor -> PIL
                images = np.array(im)            # PIL -> numpy

                mse = utils.get_MSE(images, images)  # Get MSE
                print(f'MSE: {mse}')

                # Recording outputs
                record_label_matrix[labels.item()][pred] += 1
                record_label_logit_matrix[labels.item()][pred] += top_prob.values.item()
                record_MSE_matrix[labels.item()][pred] += mse

                record_label_pred.append(pred)
                # Saving the perturbed image
                output_path = f'{path}/{model_name}/'
                os.makedirs(output_path, exist_ok=True)

        # Summary
        print('-'*50)
        print('Summary after completing on {}'.format(model_name))
        label_pred_dict = {}
        for item in record_label_pred:
            label_pred_dict[item] = label_pred_dict.get(item, 0) + 1

        # For Mean Logit: numpy -> df -> csv
        column_names = ["Original Label", "Predicted Label", "Mean Logit"]
        df_meanlogit = pd.DataFrame(columns=column_names)
        for i in range(13):
            for j in range(13):
                if (record_label_matrix[i][j] != 0):
                    record_label_logit_matrix[i][j] = record_label_logit_matrix[i][j] / float(
                        record_label_matrix[i][j])
                df_temp = pd.DataFrame({"Original Label": [utils.classes[i]],
                                        "Predicted Label": [utils.classes[j]],
                                        "Mean Logit": [record_label_logit_matrix[i][j]]})
                df_meanlogit = df_meanlogit.append(df_temp, ignore_index=True)
        df_meanlogit.to_csv(
            f'{path}/{model_name}/meanlogit.csv', index=False)

        # For Mean Prob: numpy -> df -> csv
        column_names = ["Original Label",
                        "Predicted Label", "Mean Probability"]
        meanprobability = pd.DataFrame(columns=column_names)
        for i in range(13):
            for j in range(13):
                df_temp = pd.DataFrame({"Original Label": [utils.classes[i]],
                                        "Predicted Label": [utils.classes[j]],
                                        "Mean Probability": [(record_label_logit_matrix[i][j])*100]})
                meanprobability = meanprobability.append(
                    df_temp, ignore_index=True)
        meanprobability.to_csv(
            f'{path}/{model_name}/meanprobability.csv', index=False)

        # For Predicted Counts: numpy -> df -> csv
        column_names = ["Original Label", "Predicted Label", "Count"]
        df_count = pd.DataFrame(columns=column_names)
        for i in range(13):
            for j in range(13):
                df_temp = pd.DataFrame({"Original Label": [utils.classes[i]],
                                        "Predicted Label": [utils.classes[j]],
                                        "Count": [record_label_matrix[i][j]]})
                df_count = df_count.append(df_temp, ignore_index=True)
        df_count.to_csv(
            f'{path}/{model_name}/count.csv', index=False)

        # For average MSE: numpy -> df -> csv
        column_names = ["Original Label", "Predicted Label", "Average MSE"]
        df_mse = pd.DataFrame(columns=column_names)
        for i in range(13):
            for j in range(13):
                if (record_label_matrix[i][j] != 0):
                    record_MSE_matrix[i][j] = record_MSE_matrix[i][j] / \
                        float(record_label_matrix[i][j])
                df_temp = pd.DataFrame({"Original Label": [utils.classes[i]],
                                        "Predicted Label": [utils.classes[j]],
                                        "Average MSE": [record_MSE_matrix[i][j]]})
                df_mse = df_mse.append(df_temp, ignore_index=True)
        df_mse.to_csv(
            f'{path}/{model_name}/mse.csv', index=False)


        # Heatmap
        file_name = ['meanlogit', 'meanprobability', 'count', 'mse']
        for file_name_i in file_name:
            df = pd.read_csv(
                f'{path}/{model_name}/{file_name_i}.csv')
            plt.figure(figsize=[14, 12])

            if file_name_i == 'meanlogit':
                df = df.pivot('Original Label', 'Predicted Label', 'Mean Logit')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='g', vmin=0.0, vmax=1.0)
                # sns.heatmap(hline_meanlogit, linewidths=0.8, annot=True, cmap='RdBu', vmin=0.0, vmax=1.0)
            if file_name_i == 'meanprobability':
                df = df.pivot('Original Label', 'Predicted Label', 'Mean Probability')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='g', vmin=0.0, vmax=100.0)
            if file_name_i == 'count':
                df = df.pivot('Original Label', 'Predicted Label', 'Count')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='g', annot_kws={"fontsize":12})
            if file_name_i == 'mse':
                df = df.pivot('Original Label', 'Predicted Label', 'Average MSE')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='.0f', annot_kws={"fontsize":12})

            print(
                f'Saving: {path}/{model_name}/{file_name_i}.png')
            plt.savefig(
                f'{path}/{model_name}/{file_name_i}.png')

        print('[INFO]Done')

if __name__ == '__main__':
    main()
