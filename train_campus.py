# coding=utf-8
import os
import torch
import random
import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.utils import data
from model import MLFE_net
from triplet_dataloader import TripletDataloader
from self_augmentation import Add_haze, Add_rain, GBRG2RGB, AddPepperNoise, AddGaussianNoise
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

######################################################################
# Options
# ---------------------------
# train_img = 'triplet_vgg2_train.jpg'
train_img = 'campus_train.jpg'
# loss_save_path = "./result/triplet_vgg2_loss.txt"
loss_save_path = "./result/campus_loss.txt"
batchsize = 4
poolsize = batchsize*2
learning_rate = 0.01
margin = 0.3
alpha = 0.0
gpu_id = 2
cuda_nm = 'cuda: '+str(gpu_id)
device = torch.device(cuda_nm if torch.cuda.is_available() else 'cpu')

######################################################################
# Prepare training data
# format( image_name intersection_ID location-of-intersection_ID global_location_ID )
# ---------------------------
train_data = dict()
test_data = dict()
img_nms = list()
labels = list()
with open("data_campus_train.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = line.strip('\n').split(' ')
        img_nms.append(data[0])
        labels.append(list(map(int, data[1:4])))
train_data["img_nms"] = img_nms
train_data["labels"] = np.array(labels)
img_nms = list()
labels = list()
with open("data_campus_test.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = line.strip('\n').split(' ')
        img_nms.append(data[0])
        labels.append(list(map(int, data[1:4])))
test_data["img_nms"] = img_nms
test_data["labels"] = np.array(labels)
print("{} training images will train the network! ".format(len(train_data["img_nms"])))
print("{} testing images will test the network! ".format(len(test_data["img_nms"])))

######################################################################
# Data transform
# ---------------------------
noice_transforms = transforms.Compose([

])
data_transforms = {
    'train': transforms.Compose([
        # GBRG2RGB(),
        transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224),
        # Add_haze(p=0.3),
        # Add_rain(p=0.3),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        # transforms.RandomPerspective(distortion_scale=0.05, p=0.5, interpolation=3),
        # transforms.RandomRotation(10, resample=Image.BICUBIC),
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomChoice([AddPepperNoise(p=0.3), AddGaussianNoise(p=0.3)]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        # GBRG2RGB(),
        transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
}
######################################################################
# Load Training data
# ---------------------------
image_datasets = dict()
image_datasets['train'] = TripletDataloader(train_data,
                                            data_transforms['train'])
image_datasets['val'] = TripletDataloader(test_data,
                                          data_transforms['val'])

batch = {}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

######################################################################
# Training
#---------------------------
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    # start to train
    print('\nTraining...')
    model.train()

    best_acc = 0.0
    best_epoch = 0
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_margin = 0.0
    epoch_num = 0
    data_record = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0.0
        running_margin = 0.0
        running_reg = 0.0
        epoch_num += 1
        batch_num = 0
        # Iterate over data.
        for data in dataloaders["train"]:
            # get the inputs
            inputs, labels, pos, pos_labels = data
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < batchsize:  # next epoch
                continue
            pos = pos.view(4 * batchsize, c, h, w)
            # copy pos 4times
            pos_labels = pos_labels.repeat(4).reshape(4, batchsize)
            pos_labels = pos_labels.transpose(0, 1).reshape(4 * batchsize)
            batch_num += 1

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs = Variable(inputs).to(device)
                pos = Variable(pos).to(device)
                labels = Variable(labels).to(device)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            ###############
            # forward
            # ------------- forward pass of 3 branches
            output_1 = model.forward(X=inputs, branch=1, label=None)
            output_2 = model.forward(X=inputs, branch=2, label=None)
            output_3, f = model.forward(X=inputs, branch=3, label=None)

            # ------------- calculate loss
            # branch1 loss
            loss_m = criterion(output_1[:, :5], labels[:, 0])
            loss_c = criterion(output_1[:, 5:], labels[:, 1])
            loss_br1 = loss_m + loss_c
            # print(output_2.size(), labels[:, 2])
            # branch2 loss
            loss_br2 = loss_func_2(output_2, labels[:, 2])
            # branch3 loss: Intersection ID classification
            loss_br3 = loss_func_2(output_3, labels[:, 2])

            # search hard-neg and hard-pos
            _, pos_f = model.forward(X=pos, branch=3, label=None)
            neg_labels = pos_labels

            # hard-neg
            # ----------------------------------
            nf_data = pos_f  # 4*batch * f
            # 128 is too much, we use pool size = 64
            rand = np.random.permutation(4 * batchsize)[0:poolsize]
            # print(nf_data.size(), len(np.random.permutation(4 * batchsize)))
            nf_data = nf_data[rand, :]
            neg_labels = neg_labels[rand]
            nf_t = nf_data.transpose(0, 1)  # 512*128
            score = torch.mm(f.data, nf_t)  # cosine 32*128
            score, rank = score.sort(dim=1, descending=True)  # score high == hard
            labels_cpu = labels[:, 2].cpu()
            nf_hard = torch.zeros(f.shape).to(device)
            for k in range(now_batch_size):  # find one feature with different class in pf
                hard = rank[k, :]
                for kk in hard:
                    now_label = neg_labels[kk]
                    anchor_label = labels_cpu[k]
                    if now_label != anchor_label:
                        nf_hard[k, :] = nf_data[kk, :]
                        break

            # hard-pos
            # ----------------------------------
            pf_hard = torch.zeros(f.shape).to(device)  # 32*512
            for k in range(now_batch_size):
                pf_data = pos_f[4 * k:4 * k + 4, :]
                pf_t = pf_data.transpose(0, 1)  # 512*4
                ff = f.data[k, :].reshape(1, -1)  # 1*512
                score = torch.mm(ff, pf_t)  # cosine
                score, rank = score.sort(dim=1, descending=False)  # score low == hard
                pf_hard[k, :] = pf_data[rank[0][0], :]

            # loss
            # ---------------------------------
            criterion_triplet = nn.MarginRankingLoss(margin=margin)
            pscore = torch.sum(f * pf_hard, dim=1)
            nscore = torch.sum(f * nf_hard, dim=1)
            # y = torch.ones(now_batch_size)
            # y = Variable(y.cuda())

            # loss = criterion(outputs, labels)
            # loss_triplet = criterion_triplet(f, pf, nf)
            reg = torch.sum((1 + nscore) ** 2) + torch.sum((-1 + pscore) ** 2)
            loss = torch.sum(torch.nn.functional.relu(nscore + margin - pscore))  # Here I use sum
            loss_triplet = loss + alpha * reg

            # total loss
            total_loss = 0.5 * loss_br1 + 0.5 * loss_br2 + 0.5 * loss_br3 + 1 * loss_triplet

            total_loss.backward()
            optimizer.step()

            # statistics
            running_loss += total_loss.item()
            running_corrects += float(torch.sum(pscore > nscore + margin))
            running_margin += float(torch.sum(pscore - nscore))
            running_reg += reg

            print("%d-%d, loss:%0.3f, Accuricy:%0.3f, Margin:%0.3f" % (epoch_num, batch_num,
                                                                       running_loss/(batch_num*batchsize),
                                                                       running_corrects/(batch_num*batchsize),
                                                                       running_margin/(batch_num*batchsize)))
            data_record.append([epoch_num, running_loss/(batch_num*batchsize), running_corrects/(batch_num*batchsize),
                                running_margin/(batch_num*batchsize)])
        datasize = dataset_sizes['train'] // batchsize * batchsize
        epoch_loss = running_loss / datasize
        epoch_reg = alpha * running_reg / datasize
        epoch_acc = running_corrects / datasize
        epoch_margin = running_margin / datasize

        # if epoch_acc>0.75:
        #    opt.margin = min(opt.margin+0.02, 1.0)
        print("#" * 10)
        print('now_margin: %.4f' % margin)
        print('Loss: {:.4f} Reg: {:.4f} Acc: {:.4f} MeanMargin: {:.4f}'.format(
            epoch_loss, epoch_reg, epoch_acc, epoch_margin))

        y_loss["train"].append(epoch_loss)
        y_err["train"].append(1.0 - epoch_acc)
        # deep copy the model
        if epoch_margin > best_margin:
            best_margin = epoch_margin
            save_network(model, 'vgg2_aug_best')

        if (epoch+1) % 1 == 0:
            save_network(model, epoch)
        draw_curve(epoch)

        np.savetxt(loss_save_path, data_record, fmt="%0.3f", delimiter=",", header="step, loss, mAP, margin")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


######################################################################
# FocalLoss
#---------------------------
class FocalLoss(nn.Module):
    """
    Focal loss: focus more on hard samples
    """

    def __init__(self,
                 gamma=0,
                 eps=1e-7):
        """
        :param gamma:
        :param eps:
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        :param input:
        :param target:
        :return:
        """
        log_p = self.ce(input, target)
        p = torch.exp(-log_p)
        loss = (1.0 - p) ** self.gamma * log_p
        return loss.mean()

######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="triplet_loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
#    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
#    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(train_img)


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net4campus_AUG_%s.pth' % epoch_label
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available:
        network.to(device)


######################################################################
# Load model
# ---------------------------
vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
model = MLFE_net(vgg_orig=vgg16_pretrain,
                     out_ids=68,
                     out_attribs=8).to(device)
# print(model)


# loss function
criterion = nn.CrossEntropyLoss()
loss_func_2 = FocalLoss(gamma=2).to(device)
# optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=1e-3,
                            momentum=9e-1,
                            weight_decay=1e-8)
print('=> optimize all layers.')

model = train_model(model, loss_func_2, optimizer, num_epochs=20)
