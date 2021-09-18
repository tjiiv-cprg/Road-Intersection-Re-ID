# coding=utf-8
import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('agg')
from model import MLFE_net, Single_VGG16
from self_augmentation import Add_haze, Add_rain, GBRG2RGB
import pdb

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

######################################################################
# Options
# ---------------------------
load_state = "./models/net4campus_0.pth"
curve_data_name = "./result/campus_new_detect.txt" # thres,
gpu_id = 1
cuda_nm = 'cuda: '+str(gpu_id)
device = torch.device(cuda_nm if torch.cuda.is_available() else 'cpu')

######################################################################
# Load Data
# ---------
gallery_img_nms = list()
gallery_labels = list()
with open("data_campus_gallery_trained.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = line.strip('\n').split(' ')
        gallery_img_nms.append(data[0])
        gallery_labels.append([int(data[1]), int(data[2]), int(data[3])])
print("{} images in gallery! ".format(len(gallery_img_nms)))

img_nms = list()
labels = list()
with open("data_campus_new.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = line.strip('\n').split(' ')
        img_nms.append(data[0])
        labels.append([int(data[1]), int(data[2]), int(data[3])])
print("{} testing images will test the network! ".format(len(img_nms)))

######################################################################
# Load model
# ---------------------------
print("loading model...")
vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
model = MLFE_net(vgg_orig=vgg16_pretrain, out_ids=68, out_attribs=8)
# model = Single_VGG16(vgg_orig=vgg16_pretrain, out_ids=68, out_attribs=13)
model.load_state_dict(torch.load(load_state))
model = model.eval().to(device)

######################################################################
# Extract features
# ---------------------------
print("Extracting gallery features...")
gallery_features = list()
gallery_features_int = list()
loader = transforms.Compose([#GBRG2RGB(),
                             transforms.Resize(224, interpolation=3),
                             transforms.CenterCrop(224),
                             # Add_haze(p=0.3),
                             # Add_rain(p=0.3),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
for id in range(len(gallery_img_nms)):
    img = Image.open(gallery_img_nms[id])
    img = loader(img).unsqueeze(0).to(device)
    _, f = model.forward(X=img, branch=3, label=None)
    int_det = model.forward(X=img, branch=1, label=None)
    int_det = int_det.data.cpu().squeeze(0).numpy()
    int_det = np.exp(int_det[:5])/np.sum(np.exp(int_det[:5]))
    gallery_features.append(f.data.cpu().squeeze(0).numpy().tolist())
    gallery_features_int.append(int_det.tolist())
print("%d extract features, %d vectors in each feature" % (len(gallery_features), len(gallery_features[0])))

curve_coord = np.zeros((100, 7))  # global, intersection
######################################################################
# calculate P-R
# ---------------------------
def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) * childern_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


my_list = []
for i in range(len(img_nms)):
    my_list.append(i)
my_list = list_of_groups(my_list, 4)
gallery_features = np.array(gallery_features)
gallery_features_int = np.array(gallery_features_int)
gallery_labels = np.array(gallery_labels)
labels = np.array(labels)
print("calculating each similarities...")

int_TP = 0
att_TP = 0
glo_TP = 0
total_num = 0
for id in range(len(img_nms)):
    test_img = Image.open(img_nms[id])
    test_img = loader(test_img).unsqueeze(0).to(device)
    test_b1 = model.forward(X=test_img, branch=1, label=None)
    _, test_f = model.forward(X=test_img, branch=3, label=None)
    test_b1 = test_b1.data.cpu().squeeze(0).numpy()
    test_f = test_f.data.cpu().squeeze(0).numpy()
    intersection_IDs = np.exp(test_b1[:5]) / np.sum(np.exp(test_b1[:5]))
    attribute_IDs = np.exp(test_b1[5:]) / np.sum(np.exp(test_b1[5:]))
    f_i = torch.Tensor(gallery_features).to(device)
    f_j = torch.Tensor(test_f).to(device).unsqueeze(1)
    scores_m = torch.mm(f_i, f_j).cpu().squeeze(1).numpy()
    g_int_fs = torch.Tensor(gallery_features_int).to(device)
    q_int_fs = torch.Tensor(intersection_IDs).to(device).unsqueeze(1)
    score_int = torch.mm(g_int_fs, q_int_fs).cpu().squeeze(1).numpy()

    sim_max = np.max(scores_m)
    sim_idx = np.where(sim_max == scores_m)[0][0]
    int_max = np.max(score_int)
    int_idx = np.where(int_max == score_int)[0][0]
    # int_max = np.max(intersection_IDs)
    # int_idx = np.where(int_max == intersection_IDs)[0][0]
    att_max = np.max(attribute_IDs)
    att_idx = np.where(att_max == attribute_IDs)[0][0]
    sim_thres = int(sim_max * 100)-1
    int_thres = int(int_max * 100)-1
    curve_coord[sim_thres, 0] += 1
    curve_coord[int_thres, 1] += 1
    if gallery_labels[int_idx, 0] == 0:
        curve_coord[int_thres, 2] += 1
    elif gallery_labels[int_idx, 0] == 1:
        curve_coord[int_thres, 3] += 1
    elif gallery_labels[int_idx, 0] == 2:
        curve_coord[int_thres, 4] += 1
    elif gallery_labels[int_idx, 0] == 3:
        curve_coord[int_thres, 5] += 1
    elif gallery_labels[int_idx, 0] == 4:
        curve_coord[int_thres, 6] += 1
    # for each_glo in range(len(scores_m)):
    #     curve_coord[int(scores_m[each_glo]*100)-1, 0] += 1
    # for each_int in range(len(score_int)):
    #     curve_coord[int(score_int[each_int]*100)-1, 1] += 1

    # if int_idx == labels[id][0]:
    #     int_TP += 1
    if att_idx == labels[id][1]:
        att_TP += 1
    # if labels[id][2] == gallery_labels[sim_idx][2]:
    #     glo_TP += 1
    total_num += 1
    print("int_rate: %s, att_rate: %s, glo_rate: %s" % (int_TP / total_num, att_TP / total_num, glo_TP / total_num))
# np.savetxt(curve_data_name, curve_coord, fmt="%0.0f", delimiter=" ", header="FPR, TPR, R, P")
print("congratulations! finish test!!")
