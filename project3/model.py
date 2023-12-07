import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import main
import cv2
import augmentions
import time
import math


class CNNBlock(nn.Module):
    def __init__(self, input_channels, internal_channels, kernel_size=(3, 3)):
        super(CNNBlock, self).__init__()
        self.bn = nn.BatchNorm2d(input_channels)
        padd = (math.floor(kernel_size[0] / 2), math.floor(kernel_size[1] / 2))
        self.conv1 = nn.Conv2d(input_channels, internal_channels, kernel_size, padding=padd)
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size, padding=padd)

    def forward(self, x):
        bn = self.bn(x)
        c1 = self.conv1(bn)
        c1_r = F.relu(c1)
        c2 = self.conv2(c1_r)
        c2_r = F.relu(c2)
        return F.dropout2d(c2_r, 0.4)


class Encoder(nn.Module):
    def __init__(self, args=(1, 64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.encoder_blocks = nn.ModuleList([CNNBlock(args[i], args[i + 1])
                                             for i in range(len(args) - 1)])

    def forward(self, x):
        encoded, x1 = [], x

        for b in self.encoder_blocks:
            x1 = b(x1)
            encoded.append(x1)
            x1 = self.pool(x1)
        return encoded


class Decoder(nn.Module):
    def __init__(self, args=(512, 256, 128, 64)):
        super().__init__()
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(args[i], args[i + 1], 2, 2)
                                       for i in range(len(args) - 1)])
        self.decoder_blocks = nn.ModuleList([CNNBlock(args[i], args[i + 1]) for i in range(len(args) - 1)])

    def forward(self, encoded):
        x = encoded[0]
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            decoded = self.crop(encoded[i + 1], x)
            x = torch.cat([x, decoded], dim=1)
            x = self.decoder_blocks[i](x)
        return x

    def crop(self, encoded, x):
        _, _, H, W = x.shape
        return torchvision.transforms.CenterCrop([H, W])(encoded)


def augment(X, Y, id, original_size, seed):
    # TODO add pairs of augmentations
    operation = int(original_size / (id + 1))
    np.random.seed(seed)

    if operation == 0:
        X = augmentions.dz(X, seed=seed)
        Y = augmentions.dz(Y, seed=seed)

    elif operation == 1:
        X = augmentions.dtr(X, seed=seed)
        Y = augmentions.dtr(Y, seed=seed)

    elif operation == 2:
        X = augmentions.drt(X, seed=seed)
        Y = augmentions.drt(Y, seed=seed)

    elif operation == 3:
        X = augmentions.drwt(X, seed=seed)
        Y = augmentions.drwt(Y, seed=seed)

    elif operation == 4:
        X = augmentions.dwtr(X, seed=seed)
        Y = augmentions.dwtr(Y, seed=seed)

    elif operation == 5:
        X = augmentions.dr(X, seed=seed)
        Y = augmentions.dr(Y, seed=seed)

    elif operation == 6:
        X = augmentions.ez(X, seed=seed)
        Y = augmentions.ez(Y, seed=seed)

    elif operation == 7:
        X = augmentions.ezr(X, seed=seed)
        Y = augmentions.ezr(Y, seed=seed)

    elif operation == 8:
        X = augmentions.er(X, seed=seed)
        Y = augmentions.er(Y, seed=seed)

    elif operation == 9:
        X = augmentions.trs(X, seed=seed)
        Y = augmentions.trs(Y, seed=seed)

    elif operation == 10:
        X = augmentions.dtrs(X, seed=seed)
        Y = augmentions.dtrs(Y, seed=seed)

    elif operation == 11:
        X = augmentions.rts(X, seed=seed)
        Y = augmentions.rts(Y, seed=seed)

    mean = 0
    var = 0.1
    sigma = var ** 0.5
    # Set the effect of gausian range 0 to 255...
    gaussian = np.random.normal(mean, sigma, (X.shape[0], X.shape[1])) * random.randint(30, 70)

    X = X + gaussian

    # Clip X to correct range.
    X[X < 0] = 0
    X[X > 255] = 255

    return X, Y


def show(X, Y):
    cv2.imshow("X", X)
    cv2.imshow("Y", Y.astype(np.float) * 256)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showPred(X, pred, Y, name=""):
    # zeros = np.zeros((Y.shape), dtype=np.uint8)
    # print(str(pred.shape) + " " + str(Y.shape) + " " + str(zeros.shape))
    img = np.zeros((Y.shape[1], Y.shape[2], 3))
    # print(str(max(pred)) + " " + str(max(Y)))
    img[:, :, 0] = pred[0] * 256
    img[:, :, 1] = Y[0]
    img[:, :, 2] = cv2.resize(X[0], (Y.shape[1], Y.shape[2]), interpolation=cv2.INTER_AREA) / 256
    img = cv2.resize(img, (img.shape[0] * 4, img.shape[1] * 4), interpolation=cv2.INTER_AREA)
    cv2.imshow(name, img)
    time.sleep(0.5)


class MyDataset(Dataset):
    def __init__(self, data, augment=True):
        self.data = []
        self.augment = augment
        self.seed = 0

        for video in data:
            for i in range(len(video['frames'])):
                self.data.append(
                    (video['augmented_frames'][i], video['augmented_labels'][i]))

    def __getitem__(self, idx):
        X, Y = self.data[idx % len(self.data)]

        if self.augment:
            X, Y = augment(X, Y, seed=self.seed, id=idx, original_size=len(self.data))
            self.seed += 1

        # show(X,Y)
        # show(X,Y)

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X, Y

    def __len__(self):
        return len(self.data) * (12 if self.augment else 1)


class MyTestDataset(Dataset):
    def __init__(self, data):
        self.data = []

        video_id = 0
        for video in data:
            for i in range(video['video'].shape[2]):
                self.data.append((video['augmented_frames'][i], (video['name'], video_id, i)))
            video_id += 1

    def __getitem__(self, idx):
        X = self.data[idx % len(self.data)][0]
        X = torch.tensor(X, dtype=torch.float32)
        ids = self.data[idx % len(self.data)][1]
        return X, ids

    def __len__(self):
        return len(self.data)


class UNet(nn.Module):
    def __init__(self, encoder_args, decoder_args):
        super().__init__()
        self.encoder = Encoder(encoder_args)
        self.decoder = Decoder(decoder_args)
        self.final_conv = nn.Conv2d(decoder_args[-1], 1, 1)

    def forward(self, x, out_size):
        x = x.unsqueeze(1)
        encode = self.encoder(x)
        decode = self.decoder(encode[::-1])
        res = self.final_conv(decode)
        # print(res.size())
        return res
        # return F.interpolate(res, out_size)


def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input, target):
    # Dice loss (objective to minimize) between 0 and 1
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def dice_coef_loss(y_true, y_pred):
    SMOOTHING_FACTOR = 0.0000000001
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return - (2. * intersection + SMOOTHING_FACTOR) / (y_true_f.sum() + y_pred_f.sum() + SMOOTHING_FACTOR)


def dice_coef_loss_sq(y_true, y_pred):
    SMOOTHING_FACTOR = 0.0000000001
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    t = y_true_f * y_pred_f
    intersection = (y_true_f * y_pred_f).sum()
    return - (2. * intersection + SMOOTHING_FACTOR) / (
            y_true_f.sum().square() + y_pred_f.square().sum() + SMOOTHING_FACTOR)


def focal_loss(y_true, y_pred):
    BCE_loss = nn.functional.binary_cross_entropy(y_pred, y_true, reduction="none")
    pt = torch.exp(-BCE_loss)
    return torch.mean(1 * (1 - pt) ** 2 * BCE_loss)


def combine_loss(output, Y):
    # TODO fix which loss to use (try ones from slides and their combinations)
    out_sigmoid = torch.sigmoid(output)
    loss = 0
    # loss = nn.CrossEntropyLoss(output, Y)

    # loss += (nn.functional.binary_cross_entropy(out_sigmoid, Y) * 0.2)
    # loss += dice_loss(Y, out_sigmoid)
    loss += (1 + dice_coef_loss(Y, out_sigmoid))
    # loss += focal_loss(Y, out_sigmoid)

    return loss


def validate(network, data):
    loader = DataLoader(MyDataset(data, False), batch_size=1, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    val_accuracy, val_losses = [], []
    
    network.train(False)
    network.eval()
    for it, (X, Y) in enumerate(loader):
        X, Y = (X.to(device), Y.to(device))
        output = network.forward(X, out_size=(Y.size()[1], Y.size()[1]))

        output = torch.squeeze(output, 1)
        mask = (torch.sigmoid(output) > 0.5)

        prediction = mask.cpu().detach().numpy()
        Y_cpu = Y.cpu().numpy()

        val_accuracy.append(main.validate_single_image(Y_cpu, prediction))

        loss = combine_loss(output, Y)
        val_losses.append(loss.cpu().detach().numpy())

    network.train(True)
    return sum(val_losses) / len(val_losses), np.median(val_accuracy)


def predict(network, data):
    loader = DataLoader(MyTestDataset(data), batch_size=1, num_workers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    for video in data:
        video['predictions'] = []

    network.eval()
    with torch.no_grad():
        for it, (X, (video_name, video_id, frame)) in enumerate(loader):
            X = X.to(device)
            output = network.forward(X, out_size=(X.size()[1], X.size()[2]))

            output = torch.squeeze(output, 1)
            mask = (torch.sigmoid(output) > 0.5)
            prediction = mask.cpu().detach().numpy()

            prediction = np.squeeze(prediction, 0)
            data[video_id]['predictions'].append(prediction)

    return data


def load_weights(path):
    network = UNet(encoder_args=(1, 64, 128, 256, 512, 1024),
                   decoder_args=(1024, 512, 256, 128, 64))
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    network = network.to(device)
    network.load_state_dict(torch.load(path, map_location=device))
    return network


def train(prof_train, val_train, check_point='ep-21',
          learning_rate=0.01, batch_size=16, epochs=40, print_iteration=40):
    network = UNet(encoder_args=(1, 64, 128, 256, 512, 1024),
                   decoder_args=(1024, 512, 256, 128, 64))

    # network = UNet(encoder_args=(1, 64, 128, 256),
    #                decoder_args=(256, 128, 64))

    # network = UNet(encoder_args=(1, 64, 128),
    #                decoder_args=(128, 64))

    # network = UNet(encoder_args=(1, 24, 48, 96, 192),
    #                decoder_args=(192, 96, 48, 24))

    model_save_folder = "./Trained_small_model_512_euler/signal_unet/"
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    if check_point:
        network.load_state_dict(torch.load('Trained_small_model_512_euler/signal_unet/{}.pth'.format(check_point)))
        start = int(check_point.split("-")[1]) + 1
    else:
        start = 0

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # sampler = getWeightedSampler(data_dir)
    loader = DataLoader(MyDataset(prof_train, augment=True), batch_size=batch_size, num_workers=8, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    network = network.to(device)

    # Early stopping
    last_loss = 100
    lowest_loss = last_loss
    patience = 100
    trigger_times = 0

    for epoch in range(start, epochs):
        train_losses = []
        train_accuracy = []
        for it, (X, Y) in enumerate(loader):
            X, Y = (X.to(device), Y.to(device))
            network.train(True)

            optimizer.zero_grad()
            output = network.forward(X, out_size=(Y.size()[1], Y.size()[2]))

            output = torch.squeeze(output, 1)

            mask = (torch.sigmoid(output) > 0.5)
            Y_cpu = Y.cpu().numpy()
            prediction = mask.cpu().detach().numpy()
            train_accuracy.append(main.validate_single_image(Y_cpu, prediction))

            loss = combine_loss(output, Y)
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()

            # if it == 0:
            #     showPred(X.cpu().numpy(), torch.sigmoid(output).cpu().detach().numpy(), Y_cpu,
            #              name="epoc:" + str(epoch))

            if it % print_iteration == 0:
                with torch.no_grad():
                    loss_val, acc_val = validate(network, val_train)

                    print('{:3d} / {:3d} Train loss: {:8.4f}, train accuracy: {:10.4f}, '
                        'validation loss: {:8.4f}, validation accuracy {:8.4f}'
                        .format(it, epoch, sum(train_losses) / len(train_losses),
                                sum(train_accuracy) / len(train_accuracy), loss_val, acc_val))

                    if last_loss < lowest_loss:
                        lowest_loss = last_loss
                        trigger_times = 0
                        torch.save(network.state_dict(), "{}ep-{}-{}.pth".format(model_save_folder, epoch, it))
                    else:
                        trigger_times += 1
                        if trigger_times >= patience:
                            print('Early stopping!')
                            # return network

                    if last_loss >= lowest_loss and it == 0:
                        torch.save(network.state_dict(), "{}ep-{}.pth".format(model_save_folder, epoch))

                    # if loss_val < last_loss:
                    #     torch.save(network.state_dict(), "{}ep-{}-{}.pth".format(model_save_folder, epoch, it))

                    last_loss = loss_val

                    train_losses = []
                    train_accuracy = []

                    scheduler.step(loss_val)

        if epoch % 10 == 0:
            cv2.destroyAllWindows()

        # torch.save(network.state_dict(), "{}ep-{}.pth".format(model_save_folder, epoch))

    torch.save(network.state_dict(), "{}Final.pth".format(model_save_folder))
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
