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
from self_augmentation import GBRG2RGB

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

######################################################################
# Options
# ---------------------------
data_dir = "***/robotcar"
load_state = './models/2B_TRI_AUG.pth'
curve_data_name = "./result/PR_2B_FOC.txt"
gpu_id = 1
cuda_nm = 'cuda: '+str(gpu_id)
device = torch.device(cuda_nm if torch.cuda.is_available() else 'cpu')

######################################################################
# Load Data
# ---------
gallery_img_nms = list()
gallery_labels = list()
with open("gallery_imgs.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = line.strip('\n').split(' ')
        gallery_img_nms.append(data[0])
        gallery_labels.append(int(data[3]))
print("{} images in gallery! ".format(len(gallery_img_nms)))

img_nms = list()
labels = list()
with open("data_robotcar_test_hightraffic.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = line.strip('\n').split(' ')
        img_nms.append(data[0])
        labels.append(int(data[3]))
print("{} testing images will test the network! ".format(len(img_nms)))

######################################################################
# Load model
# ---------------------------
print("loading model...")
vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
model = MLFE_net(vgg_orig=vgg16_pretrain, out_ids=68, out_attribs=13)
# model = Single_VGG16(vgg_orig=vgg16_pretrain, out_ids=68, out_attribs=13)
model.load_state_dict(torch.load(load_state))
model = model.eval().to(device)

######################################################################
# Extract features
# ---------------------------
print("Extracting gallery features...")
gallery_features = list()
loader = transforms.Compose([GBRG2RGB(),
                             transforms.Resize(224, interpolation=3),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
for id in range(len(gallery_img_nms)):
    img = Image.open(gallery_img_nms[id])
    img = loader(img).unsqueeze(0).to(device)
    _, f = model.forward(X=img, branch=3, label=None)
    gallery_features.append(f.data.cpu().squeeze(0).numpy().tolist())
print("%d extract features, %d vectors in each feature" % (len(gallery_features), len(gallery_features[0])))

curve_coord = np.zeros((100, 4))  # t_p, f_p, t_n, f_n
final_rate = np.zeros(2)
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
gallery_labels = np.array(gallery_labels)
labels = np.array(labels)
print("calculating each similarities...")

for list_i in my_list:
    if len(list_i) < 4:
        continue
    test_img0 = Image.open(img_nms[list_i[0]])
    test_img1 = Image.open(img_nms[list_i[1]])
    test_img2 = Image.open(img_nms[list_i[2]])
    test_img3 = Image.open(img_nms[list_i[3]])
    test_img0 = loader(test_img0)
    test_img1 = loader(test_img1)
    test_img2 = loader(test_img2)
    test_img3 = loader(test_img3)
    c, h, w = test_img0.shape
    test_imgs = torch.cat((test_img0.view(1, c, h, w), test_img1.view(1, c, h, w),
                          test_img2.view(1, c, h, w), test_img3.view(1, c, h, w)), 0).to(device)
    _, test_f = model.forward(X=test_imgs, branch=3, label=None)
    test_f = test_f.data.cpu().numpy()

    f_i = torch.Tensor(gallery_features).to(device)
    f_j = torch.Tensor(test_f).to(device).transpose(0, 1)
    scores_m = torch.mm(f_i, f_j).cpu().numpy()
    scores = scores_m.flatten()

    label_i = np.array(gallery_labels)[:, np.newaxis]
    mat_i = np.repeat(label_i, len(list_i), axis=1).flatten()
    label_j = np.array(labels[list_i])[np.newaxis, :]
    mat_j = np.repeat(label_j, len(gallery_labels), axis=0).flatten()

    for thres_t in range(0, 100):
        thres = 1 - thres_t / 100
        p_pred = np.argwhere(scores >= thres)
        p_label = np.argwhere(mat_i == mat_j)
        curve_coord[thres_t, 1] += len(np.setdiff1d(p_pred, p_label))
        curve_coord[thres_t, 0] += len(np.intersect1d(p_pred, p_label))

        n_pred = np.argwhere(scores < thres)
        n_label = np.argwhere(mat_i != mat_j)
        curve_coord[thres_t, 3] += len(np.setdiff1d(n_pred, n_label))
        curve_coord[thres_t, 2] += len(np.intersect1d(n_pred, n_label))

    result_index = np.where(scores_m[:, 0] == np.max(scores_m[:, 0]))[0][0]
    if gallery_labels[result_index] == labels[list_i[0]]:
        final_rate[0] += 1
    result_index = np.where(scores_m[:, 1] == np.max(scores_m[:, 1]))[0][0]
    if gallery_labels[result_index] == labels[list_i[1]]:
        final_rate[0] += 1
    result_index = np.where(scores_m[:, 2] == np.max(scores_m[:, 2]))[0][0]
    if gallery_labels[result_index] == labels[list_i[2]]:
        final_rate[0] += 1
    result_index = np.where(scores_m[:, 3] == np.max(scores_m[:, 3]))[0][0]
    if gallery_labels[result_index] == labels[list_i[3]]:
        final_rate[0] += 1
    final_rate[1] += 4
    print("Num: %d, Acc: %0.3f (%d, %d)" % (list_i[3], final_rate[0]/final_rate[1], final_rate[0], final_rate[1]))

# t_p, f_p, t_n, f_n
curve_data = np.zeros((100, 4))  # FPR,TPR,P,R
curve_data[:, 0] = curve_coord[:, 1] / (curve_coord[:, 1] + curve_coord[:, 2] + 0.0001)
curve_data[:, 1] = curve_coord[:, 0] / (curve_coord[:, 0] + curve_coord[:, 3] + 0.0001)
curve_data[:, 2] = curve_coord[:, 0] / (curve_coord[:, 0] + curve_coord[:, 1] + 0.0001)
curve_data[:, 3] = curve_coord[:, 0] / (curve_coord[:, 0] + curve_coord[:, 3] + 0.0001)

print("saving results...")
# np.savetxt(curve_data_name, curve_data, fmt="%0.3f", delimiter=" ", header="FPR, TPR, R, P")

print("congratulations! finish test!!")
print("FPR:%0.3f, TPR:%0.3f, R:%0.3f, P:%0.3f" % (curve_data[50][0],
                                                  curve_data[50][1],
                                                  curve_data[50][2],
                                                  curve_data[50][3]))


