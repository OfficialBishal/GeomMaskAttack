import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'F')


def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A


def clip(x): return clip_tensor(x, 0, 255)


def imshow_transform(img, labels):
    tf = transforms.Compose([transforms.Lambda(clip),
                            transforms.ToPILImage()])
    plt.figure()
    plt.imshow(tf(img.cpu()[0]))
    plt.title(labels)
    plt.show()


def classification_prediction(model, images):
    with torch.no_grad():
        output = model(images)

    probabilities = torch.nn.Softmax(dim=-1)(output)
    sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
    predictions = np.argmax(list(probabilities.numpy()), axis=1)

    # loop over the predictions and display the rank-13 predictions and
    # corresponding probabilities to our terminal
    for (i, idx) in enumerate(sortedProba[0, :13]):
        print("{}: {:.10f}%".format(
            classes[idx.item()].strip(), probabilities[0, idx.item()] * 100))
    # print(sortedProba[0,0].numpy())
    # print('Predictions: ', predictions)


# Conversion: numpy (160, 105, 3) -> PIL -> tensor [1, 3, 160, 105]
def numpy_PIL_tensor(data):
    PIL_image = Image.fromarray(np.uint8(data)).convert('RGB')
    transform_ori = transforms.Compose([transforms.ToTensor()])

    x = transform_ori(PIL_image)
    x.unsqueeze_(0)     # [3, 160, 105] -> [1, 3, 160, 105]
    return x


# Returns logit and index of top 1 prediction.
def get_confidence(model, data):
    x = numpy_PIL_tensor(data)
    output = model(x)

    sm = torch.nn.Softmax()
    probabilities = torch.nn.Softmax(dim=-1)(output)
    top1_prob = torch.topk(probabilities, 1)

    return top1_prob


def get_MSE(original_img, perturbed_img):
    n = 160*105
    error = 0.0
    for d in range(3):
        for i in range(160):
            for j in range(105):
                error += (original_img[i, j, d] - perturbed_img[i, j, d])**2
    error = error/(n*3)
    return error
