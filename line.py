# sopt.py

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
import torchvision.transforms as transforms
import utils


def get_rgb(img_in):
    sample = img_in.copy()
    rgb = dict()
    rgb['r'] = sample[0, 0, 0]
    rgb['g'] = sample[0, 0, 1]
    rgb['b'] = sample[0, 0, 2]
    distance = np.sqrt(sample[0, 0, 0]**2 +
                       sample[0, 0, 1]**2 + sample[0, 0, 2]**2)

    for i in range(160):
        for j in range(105):
            distance_temp = np.sqrt(
                sample[i, j, 0]**2 + sample[i, j, 1]**2 + sample[i, j, 2]**2)
            if distance_temp < distance:
                distance = distance_temp
                rgb['r'] = sample[i, j, 0]
                rgb['g'] = sample[i, j, 1]
                rgb['b'] = sample[i, j, 2]
    # print(rgb)
    return rgb


def create_vline(img_in, x, thickness, rgb):

    spot = np.full((160, 105), False, dtype=bool)
    for i in range(x, x+thickness, 1):
        spot[:, i] = True

    sample = img_in.copy()

    sample[spot, 0] = rgb['r']
    sample[spot, 1] = rgb['g']
    sample[spot, 2] = rgb['b']

    return sample   # shape: (160, 105, 3)


def create_hline(img_in, y, thickness, rgb):

    spot = np.full((160, 105), False, dtype=bool)
    for i in range(y, y+thickness, 1):
        spot[i, :] = True

    sample = img_in.copy()

    sample[spot, 0] = rgb['r']
    sample[spot, 1] = rgb['g']
    sample[spot, 2] = rgb['b']

    return sample   # shape: (160, 105, 3)


# Conversion: numpy (160, 105, 3) -> PIL -> tensor [1, 3, 160, 105]
def numpy_PIL_tensor(data):
    PIL_image = Image.fromarray(np.uint8(data)).convert('RGB')
    transform_ori = transforms.Compose([transforms.ToTensor()])

    x = transform_ori(PIL_image)
    x.unsqueeze_(0)     # [3, 160, 105] -> [1, 3, 160, 105]
    return x


# Image size = 160x105
def get_prediction(model, data):

    x = numpy_PIL_tensor(data)

    output = model(x)
    _, predicted = torch.max(output, 1)
    return (predicted.item(), output)


def main(model, loss_fn, im, labels, vline, hline, thickness):

    # im = im.permute(1, 2, 0)    # tensor[C,H,W] -> tensor[H,W,C]
    im = transforms.ToPILImage()(im).convert("RGB")  # tensor -> PIL
    img = np.array(im)            # PIL -> numpy
    # print('img',img.shape)
    # print(img)
    rgb = get_rgb(img)
    label = labels.item()   # tensor[labels] -> int(labels)
    loss_temp = 0
    y, x = img.shape[:2]
    hit = 0     # counter for number to times misclassification was successful

    if (vline == True and hline == False):
        for i in range(x-thickness+1):
            adversarial = create_vline(img, i, thickness, rgb)
            pred, output = get_prediction(model, adversarial)
            loss = loss_fn(output, labels)
            if(pred != label):      # Misclassfification successful
                if(loss.item() > loss_temp):    # Has more loss (good misclassification)
                    attack = (i, pred, loss)
                    loss_temp = loss.item()
                    hit += 1
            

    if (vline == False and hline == True):
        for i in range(y-thickness+1):
            adversarial = create_hline(img, i, thickness, rgb)
            pred, output = get_prediction(model, adversarial)
            loss = loss_fn(output, labels)
            if(pred != label):      # Misclassfification successful
                if(loss.item() > loss_temp):    # Has more loss (good misclassification)
                    attack = (i, pred, loss)
                    loss_temp = loss.item()
                    hit += 1


    if (hit != 0):
        print("[INFO]Successfull")
        success = True
        line, label_pert, loss = attack
        if (vline == True and hline == False):
            pert_img = create_vline(img, line, thickness, rgb)
        if (vline == False and hline == True):
            pert_img = create_hline(img, line, thickness, rgb)

        # Get probability of perturbed label
        top_prob = utils.get_confidence(model, pert_img)
        print(f'Prob: {top_prob.values.item()}, Index: {top_prob.indices.item()}')

        mse = utils.get_MSE(img, pert_img) # Get MSE
        print(f'MSE: {mse}')

        print(f'Total hits: {hit}')
        pert_img = numpy_PIL_tensor(pert_img)

        return pert_img, label_pert, success, top_prob, mse

    elif (hit == 0):
        success = False
        top_prob = utils.get_confidence(model, img)
        img = numpy_PIL_tensor(img)
        return img, label, success, top_prob, 0.0
