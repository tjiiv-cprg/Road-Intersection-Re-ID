import numpy as np
import torch
from PIL import Image


class TripletDataloader(torch.utils.data.Dataset):

    def __init__(self, input_data, transform):
        self.img_nms = input_data["img_nms"]
        self.labels = input_data["labels"]
        self.transform = transform

    def _get_pos_sample(self, index):  # randomly select 4 samples in same class samples but different from index
        all_ids = self.labels[:, 2]
        index_id = self.labels[index, 2]
        pos_index = np.argwhere(all_ids == index_id)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        pos_nms = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            pos_nms.append(self.img_nms[tmp_index])
        return pos_nms

    def __getitem__(self, index):
        img = Image.open(self.img_nms[index])
        label = self.labels[index]
        pos_nms = self._get_pos_sample(index)
        pos_img0 = Image.open(pos_nms[0])
        pos_img1 = Image.open(pos_nms[1])
        pos_img2 = Image.open(pos_nms[2])
        pos_img3 = Image.open(pos_nms[3])
        if self.transform is not None:
            img = self.transform(img)
            pos_img0 = self.transform(pos_img0)
            pos_img1 = self.transform(pos_img1)
            pos_img2 = self.transform(pos_img2)
            pos_img3 = self.transform(pos_img3)
        c, h, w = pos_img0.shape
        pos_imgs = torch.cat((pos_img0.view(1, c, h, w), pos_img1.view(1, c, h, w),
                              pos_img2.view(1, c, h, w), pos_img3.view(1, c, h, w)), 0)
        pos_label = label[2]
        return img, label, pos_imgs, pos_label


    def __len__(self):
        return len(self.img_nms)
