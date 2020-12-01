import torch
import torch.nn as nn
import math
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import copy
from collections import OrderedDict


class MLFE_net(nn.Module):

    def __init__(self, vgg_orig, out_ids, out_attribs):
        super(MLFE_net, self).__init__()

        self.out_ids, self.out_attribs = out_ids, out_attribs
        print('=> out_ids: %d, out_attribs: %d' % (self.out_ids, self.out_attribs))

        feats = vgg_orig.features._modules
        classifier = vgg_orig.classifier._modules

        # Conv1
        self.conv1_1 = copy.deepcopy(feats['0'])  # (0)
        self.conv1_2 = copy.deepcopy(feats['1'])  # (1)
        self.conv1_3 = copy.deepcopy(feats['2'])  # (2)
        self.conv1_4 = copy.deepcopy(feats['3'])  # (3)
        self.conv1_5 = copy.deepcopy(feats['4'])  # (4)

        self.conv1 = torch.nn.Sequential(OrderedDict([
            ("conv1", self.conv1_1),
            ("relu1", self.conv1_2),
            ("conv2", self.conv1_3),
            ("relu2", self.conv1_4),
            ("maxpool1", self.conv1_5)
        ]))

        # Conv2
        self.conv2_1 = copy.deepcopy(feats['5'])  # (5)
        self.conv2_2 = copy.deepcopy(feats['6'])  # (6)
        self.conv2_3 = copy.deepcopy(feats['7'])  # (7)
        self.conv2_4 = copy.deepcopy(feats['8'])  # (8)
        self.conv2_5 = copy.deepcopy(feats['9'])  # (9)

        self.conv2 = torch.nn.Sequential(OrderedDict([
            ("conv3", self.conv2_1),
            ("relu3", self.conv2_2),
            ("conv4", self.conv2_3),
            ("relu4", self.conv2_4),
            ("maxpool2", self.conv2_5)
        ]))

        # Conv3
        self.conv3_1 = copy.deepcopy(feats['10'])  # (10)
        self.conv3_2 = copy.deepcopy(feats['11'])  # (11)
        self.conv3_3 = copy.deepcopy(feats['12'])  # (12)
        self.conv3_4 = copy.deepcopy(feats['13'])  # (13)
        self.conv3_5 = copy.deepcopy(feats['14'])  # (14)
        self.conv3_6 = copy.deepcopy(feats['15'])  # (15)
        self.conv3_7 = copy.deepcopy(feats['16'])  # (16)

        self.conv3 = torch.nn.Sequential(OrderedDict([
            ("conv5", self.conv3_1),
            ("relu5", self.conv3_2),
            ("conv6", self.conv3_3),
            ("relu6", self.conv3_4),
            ("conv7", self.conv3_5),
            ("relu7", self.conv3_6),
            ("maxpool3", self.conv3_7)
        ]))

        # Conv4_1
        self.conv4_1_1 = copy.deepcopy(feats['17'])  # (17)
        self.conv4_1_2 = copy.deepcopy(feats['18'])  # (18)
        self.conv4_1_3 = copy.deepcopy(feats['19'])  # (19)
        self.conv4_1_4 = copy.deepcopy(feats['20'])  # (20)
        self.conv4_1_5 = copy.deepcopy(feats['21'])  # (21)
        self.conv4_1_6 = copy.deepcopy(feats['22'])  # (22)
        self.conv4_1_7 = copy.deepcopy(feats['23'])  # (23)

        self.conv4_1 = torch.nn.Sequential(OrderedDict([
            ("conv8_1", self.conv4_1_1),
            ("relu8_1", self.conv4_1_2),
            ("conv9_1", self.conv4_1_3),
            ("relu9_1", self.conv4_1_4),
            ("conv10_1", self.conv4_1_5),
            ("relu10_1", self.conv4_1_6),
            ("maxpool4_1", self.conv4_1_7)
        ]))

        # Conv4_2
        self.conv4_2_1 = copy.deepcopy(self.conv4_1_1)
        self.conv4_2_2 = copy.deepcopy(self.conv4_1_2)
        self.conv4_2_3 = copy.deepcopy(self.conv4_1_3)
        self.conv4_2_4 = copy.deepcopy(self.conv4_1_4)
        self.conv4_2_5 = copy.deepcopy(self.conv4_1_5)
        self.conv4_2_6 = copy.deepcopy(self.conv4_1_6)
        self.conv4_2_7 = copy.deepcopy(self.conv4_1_7)

        self.conv4_2 = torch.nn.Sequential(OrderedDict([
            ("conv8_2", self.conv4_2_1),
            ("relu8_2", self.conv4_2_2),
            ("conv9_2", self.conv4_2_3),
            ("relu9_2", self.conv4_2_4),
            ("conv10_2", self.conv4_2_5),
            ("relu10_2", self.conv4_2_6),
            ("maxpool4_2", self.conv4_2_7)
        ]))

        # Conv5_1
        self.conv5_1_1 = copy.deepcopy(feats['24'])  # (24)
        self.conv5_1_2 = copy.deepcopy(feats['25'])  # (25)
        self.conv5_1_3 = copy.deepcopy(feats['26'])  # (26)
        self.conv5_1_4 = copy.deepcopy(feats['27'])  # (27)
        self.conv5_1_5 = copy.deepcopy(feats['28'])  # (28)
        self.conv5_1_6 = copy.deepcopy(feats['29'])  # (29)
        self.conv5_1_7 = copy.deepcopy(feats['30'])  # (30)

        self.conv5_1 = torch.nn.Sequential(OrderedDict([
            ("conv11_1", self.conv5_1_1),
            ("relu11_1", self.conv5_1_2),
            ("conv12_1", self.conv5_1_3),
            ("relu12_1", self.conv5_1_4),
            ("conv13_1", self.conv5_1_5),
            ("relu13_1", self.conv5_1_6),
            ("maxpool5_1", self.conv5_1_7)
        ]))

        # Conv5_2
        self.conv5_2_1 = copy.deepcopy(self.conv5_1_1)
        self.conv5_2_2 = copy.deepcopy(self.conv5_1_2)
        self.conv5_2_3 = copy.deepcopy(self.conv5_1_3)
        self.conv5_2_4 = copy.deepcopy(self.conv5_1_4)
        self.conv5_2_5 = copy.deepcopy(self.conv5_1_5)
        self.conv5_2_6 = copy.deepcopy(self.conv5_1_6)
        self.conv5_2_7 = copy.deepcopy(self.conv5_1_7)

        self.conv5_2 = torch.nn.Sequential(OrderedDict([
            ("conv11_2", self.conv5_2_1),
            ("relu11_2", self.conv5_2_2),
            ("conv12_2", self.conv5_2_3),
            ("relu12_2", self.conv5_2_4),
            ("conv13_2", self.conv5_2_5),
            ("relu13_2", self.conv5_2_6),
            ("maxpool5_2", self.conv5_2_7)
        ]))

        # FC6_1
        self.FC6_1_1 = copy.deepcopy(classifier['0'])  # (0)
        self.FC6_1_2 = copy.deepcopy(classifier['1'])  # (1)
        self.FC6_1_3 = copy.deepcopy(classifier['2'])  # (2)
        self.FC6_1_4 = copy.deepcopy(classifier['3'])  # (3)
        self.FC6_1_5 = copy.deepcopy(classifier['4'])  # (4)
        self.FC6_1_6 = copy.deepcopy(classifier['5'])  # (5)

        self.FC6_1 = torch.nn.Sequential(OrderedDict([
            ("linear1_1", self.FC6_1_1),
            ("relu14_1", self.FC6_1_2),
            ("dropout1_1", self.FC6_1_3),
            ("linear2_1", self.FC6_1_4),
            ("relu15_1", self.FC6_1_5),
            ("dropout2_1", self.FC6_1_6)
        ]))

        # FC6_2
        self.FC6_2_1 = copy.deepcopy(self.FC6_1_1)
        self.FC6_2_2 = copy.deepcopy(self.FC6_1_2)
        self.FC6_2_3 = copy.deepcopy(self.FC6_1_3)
        self.FC6_2_4 = copy.deepcopy(self.FC6_1_4)
        self.FC6_2_5 = copy.deepcopy(self.FC6_1_5)
        self.FC6_2_6 = copy.deepcopy(self.FC6_1_6)

        self.FC6_2 = torch.nn.Sequential(OrderedDict([
            ("linear1_2", self.FC6_2_1),
            ("relu14_2", self.FC6_2_2),
            ("dropout1_2", self.FC6_2_3),
            ("linear2_2", self.FC6_2_4),
            ("relu15_2", self.FC6_2_5),
            ("dropout2_2", self.FC6_2_6)
        ]))

        # FC7_1
        self.FC7_1 = copy.deepcopy(classifier['6'])  # (6): 4096, 1000

        # FC7_2
        self.FC7_2 = copy.deepcopy(self.FC7_1)

        # ------------------------------ extra layers: FC8 and FC9
        self.FC_8 = torch.nn.Linear(in_features=2000,  # 2000
                                    out_features=1000)  # 1000

        # final output layer: intersection id classifier: using arc_fc_br2
        self.coarse_classifier = torch.nn.Linear(in_features=1000,
                                                 out_features=out_ids)

        # attribute classifiers: out_attribs to be decided
        self.attrib_classifier = torch.nn.Linear(in_features=1000,
                                                 out_features=out_attribs)

        # final classifiers: out_ids to be decided
        self.final_classifier = torch.nn.Linear(in_features=1000,
                                                out_features=out_ids)


        # Arc FC layer for branch_2 and branch_3
        # self.arc_fc_br2 = ArcFC(in_features=1000,
        #                         out_features=out_ids,
        #                         s=30.0,
        #                         m=0.5,
        #                         easy_margin=False)
        # self.arc_fc_br3 = ArcFC(in_features=1024,
        #                         out_features=out_ids,
        #                         s=30.0,
        #                         m=0.5,
        #                         easy_margin=False)

        # 构建分支
        self.shared_layers = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.branch_1_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_1,
            self.conv5_1,
        )

        self.branch_1_fc = torch.nn.Sequential(
            self.FC6_1,
            self.FC7_1
        )

        self.branch_1 = torch.nn.Sequential(
            self.branch_1_feats,
            self.branch_1_fc
        )

        self.branch_2_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_2,
            self.conv5_2
        )

        self.branch_2_fc = torch.nn.Sequential(
            self.FC6_2,
            self.FC7_2
        )

        self.branch_2 = torch.nn.Sequential(
            self.branch_2_feats,
            self.branch_2_fc
        )
        # model_ft = models.resnet50(pretrained=True)
        # # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.model = model_ft
        # self.classifier = ClassBlock(2048, class_num, dropout=False, relu=False)

    def forward(self, X, branch, label=None):
        """
        先单独训练branch_1, 然后brach_1, branch_2, branch_3联合训练
        :param X:
        :param branch:
        :param label:
        :return:
        """
        # batch size
        N = X.size(0)

        if branch == 1:  # train attributes classification
            X = self.branch_1_feats(X)

            # reshape and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_1_fc(X)

            assert X.size() == (N, 1000)

            X = self.attrib_classifier(X)

            assert X.size() == (N, self.out_attribs)

            return X

        elif branch == 2:  # get intersection fine-grained feature
            # if label is None:
            #     print('=> label is None.')
            #     return None
            X = self.branch_2_feats(X)

            # reshape and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_2_fc(X)

            assert X.size() == (N, 1000)

            X = self.coarse_classifier(X)

            assert X.size() == (N, self.out_ids)

            return X

        elif branch == 3:  # overall: combine branch_1 and branch_2
            # if label is None:
            #     print('=> label is None.')
            #     return None
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)

            # reshape and connect to FC layers
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)

            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)

            assert fusion_feats.size() == (N, 2000)
            # connect to FC8: output 1000 dim feature vector
            final_feature = self.FC_8(fusion_feats)

            assert final_feature.size() == (N, 1000)
            f_norm = final_feature.norm(p=2, dim=1, keepdim=True) + 1e-8
            final_feature = final_feature.div(f_norm)

            X = self.final_classifier(final_feature)

            assert X.size() == (N, self.out_ids)

            return X, final_feature

        elif branch == 4:  # test coarse classifier
            # extract features
            X = self.branch_1_feats(X)

            # flatten and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_1_fc(X)

            assert X.size() == (N, 1000)

            X = self.attrib_classifier(X)

            assert X.size() == (N, self.out_attribs)

            return X

        elif branch == 5:  # test final classifier
            # 前向运算提取用于Intersection ID的特征向量
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)

            # reshape and connect to FC layers
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)

            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)

            # feature fusion
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)

            assert fusion_feats.size() == (N, 2000)

            # connect to FC8: output 1000 dim feature vector
            final_feature = self.FC_8(fusion_feats)

            assert final_feature.size() == (N, 1000)

            X = self.final_classifier(final_feature)

            assert X.size() == (N, self.out_ids)

            return X, final_feature

        else:
            print('=> invalid branch')
            return None
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = torch.squeeze(x)
        # x,f = self.classifier(x)
        # return x,f


class Single_VGG16(nn.Module):

    def __init__(self, vgg_orig, out_ids, out_attribs):
        super(Single_VGG16, self).__init__()

        self.out_ids, self.out_attribs = out_ids, out_attribs
        print('=> out_ids: %d, out_attribs: %d' % (self.out_ids, self.out_attribs))

        feats = vgg_orig.features._modules
        classifier = vgg_orig.classifier._modules

        # Conv1
        self.conv1_1 = copy.deepcopy(feats['0'])  # (0)
        self.conv1_2 = copy.deepcopy(feats['1'])  # (1)
        self.conv1_3 = copy.deepcopy(feats['2'])  # (2)
        self.conv1_4 = copy.deepcopy(feats['3'])  # (3)
        self.conv1_5 = copy.deepcopy(feats['4'])  # (4)

        self.conv1 = torch.nn.Sequential(OrderedDict([
            ("conv1", self.conv1_1),
            ("relu1", self.conv1_2),
            ("conv2", self.conv1_3),
            ("relu2", self.conv1_4),
            ("maxpool1", self.conv1_5)
        ]))

        # Conv2
        self.conv2_1 = copy.deepcopy(feats['5'])  # (5)
        self.conv2_2 = copy.deepcopy(feats['6'])  # (6)
        self.conv2_3 = copy.deepcopy(feats['7'])  # (7)
        self.conv2_4 = copy.deepcopy(feats['8'])  # (8)
        self.conv2_5 = copy.deepcopy(feats['9'])  # (9)

        self.conv2 = torch.nn.Sequential(OrderedDict([
            ("conv3", self.conv2_1),
            ("relu3", self.conv2_2),
            ("conv4", self.conv2_3),
            ("relu4", self.conv2_4),
            ("maxpool2", self.conv2_5)
        ]))

        # Conv3
        self.conv3_1 = copy.deepcopy(feats['10'])  # (10)
        self.conv3_2 = copy.deepcopy(feats['11'])  # (11)
        self.conv3_3 = copy.deepcopy(feats['12'])  # (12)
        self.conv3_4 = copy.deepcopy(feats['13'])  # (13)
        self.conv3_5 = copy.deepcopy(feats['14'])  # (14)
        self.conv3_6 = copy.deepcopy(feats['15'])  # (15)
        self.conv3_7 = copy.deepcopy(feats['16'])  # (16)

        self.conv3 = torch.nn.Sequential(OrderedDict([
            ("conv5", self.conv3_1),
            ("relu5", self.conv3_2),
            ("conv6", self.conv3_3),
            ("relu6", self.conv3_4),
            ("conv7", self.conv3_5),
            ("relu7", self.conv3_6),
            ("maxpool3", self.conv3_7)
        ]))

        # Conv4_2
        self.conv4_2_1 = copy.deepcopy(feats['17'])  # (17)
        self.conv4_2_2 = copy.deepcopy(feats['18'])  # (18)
        self.conv4_2_3 = copy.deepcopy(feats['19'])  # (19)
        self.conv4_2_4 = copy.deepcopy(feats['20'])  # (20)
        self.conv4_2_5 = copy.deepcopy(feats['21'])  # (21)
        self.conv4_2_6 = copy.deepcopy(feats['22'])  # (22)
        self.conv4_2_7 = copy.deepcopy(feats['23'])  # (23)

        self.conv4_2 = torch.nn.Sequential(OrderedDict([
            ("conv8_2", self.conv4_2_1),
            ("relu8_2", self.conv4_2_2),
            ("conv9_2", self.conv4_2_3),
            ("relu9_2", self.conv4_2_4),
            ("conv10_2", self.conv4_2_5),
            ("relu10_2", self.conv4_2_6),
            ("maxpool4_2", self.conv4_2_7)
        ]))

        # Conv5_2
        self.conv5_2_1 = copy.deepcopy(feats['24'])  # (24)
        self.conv5_2_2 = copy.deepcopy(feats['25'])  # (25)
        self.conv5_2_3 = copy.deepcopy(feats['26'])  # (26)
        self.conv5_2_4 = copy.deepcopy(feats['27'])  # (27)
        self.conv5_2_5 = copy.deepcopy(feats['28'])  # (28)
        self.conv5_2_6 = copy.deepcopy(feats['29'])  # (29)
        self.conv5_2_7 = copy.deepcopy(feats['30'])  # (30)

        self.conv5_2 = torch.nn.Sequential(OrderedDict([
            ("conv11_2", self.conv5_2_1),
            ("relu11_2", self.conv5_2_2),
            ("conv12_2", self.conv5_2_3),
            ("relu12_2", self.conv5_2_4),
            ("conv13_2", self.conv5_2_5),
            ("relu13_2", self.conv5_2_6),
            ("maxpool5_2", self.conv5_2_7)
        ]))

        # FC6_2
        self.FC6_2_1 = copy.deepcopy(classifier['0'])  # (0)
        self.FC6_2_2 = copy.deepcopy(classifier['1'])  # (1)
        self.FC6_2_3 = copy.deepcopy(classifier['2'])  # (2)
        self.FC6_2_4 = copy.deepcopy(classifier['3'])  # (3)
        self.FC6_2_5 = copy.deepcopy(classifier['4'])  # (4)
        self.FC6_2_6 = copy.deepcopy(classifier['5'])  # (5)

        self.FC6_2 = torch.nn.Sequential(OrderedDict([
            ("linear1_2", self.FC6_2_1),
            ("relu14_2", self.FC6_2_2),
            ("dropout1_2", self.FC6_2_3),
            ("linear2_2", self.FC6_2_4),
            ("relu15_2", self.FC6_2_5),
            ("dropout2_2", self.FC6_2_6)
        ]))

        # FC7_2
        self.FC7_2 = copy.deepcopy(classifier['6'])  # (6): 4096, 1000

        # ------------------------------ extra layers: FC8 and FC9
        # self.FC_8 = torch.nn.Linear(in_features=1000,  # 2000
        #                             out_features=1000)  # 1000

        # final output layer: intersection id classifier: using arc_fc_br2
        # self.coarse_classifier = torch.nn.Linear(in_features=1000,
        #                                          out_features=out_ids)

        # final classifiers: out_ids to be decided
        self.final_classifier = torch.nn.Linear(in_features=1000,
                                                out_features=out_ids)

        # 构建分支
        self.shared_layers = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.branch_2_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_2,
            self.conv5_2
        )

        self.branch_2_fc = torch.nn.Sequential(
            self.FC6_2,
            self.FC7_2
        )

    def forward(self, X, branch, label=None):
        """
        先单独训练branch_1, 然后brach_1, branch_2, branch_3联合训练
        :param X:
        :param branch:
        :param label:
        :return:
        """
        # batch size
        N = X.size(0)

        if branch == 1:  # train attributes classification
            print("branch 1 was canceled!")
            return 0

        elif branch == 2:  # get intersection fine-grained feature

            X = self.branch_2_feats(X)

            # reshape and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_2_fc(X)

            assert X.size() == (N, 1000)

            X = self.coarse_classifier(X)

            assert X.size() == (N, self.out_ids)

            return X

        elif branch == 3:  # overall: combine branch_1 and branch_2
            branch_2 = self.branch_2_feats(X)
            branch_2 = branch_2.view(N, -1)
            final_feature = self.branch_2_fc(branch_2)

            assert final_feature.size() == (N, 1000)
            # connect to FC8: output 1000 dim feature vector
            # final_feature = self.FC_8(branch_2)

            # assert final_feature.size() == (N, 1000)
            f_norm = final_feature.norm(p=2, dim=1, keepdim=True) + 1e-8
            final_feature = final_feature.div(f_norm)

            X = self.final_classifier(final_feature)

            assert X.size() == (N, self.out_ids)

            return X, final_feature

        else:
            print('=> invalid branch')
            return None

