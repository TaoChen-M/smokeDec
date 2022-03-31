import torch
import time
import os
from torch import nn

from PIL import Image
import numpy as np
from torchvision import models
from torchvision import transforms as T


class Model:
    def __init__(self, name):
        self.do_use_cuda = torch.cuda.is_available()
        self.name = name
        print("create model named:{}".format(name))
        # 加载模型，不使用预训练模型
        self.backbone = models.__dict__[name](pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 2)
        )
        # self.backbone = nn.DataParallel(self.backbone)

        # 现在使用的高斯噪声是ImageNet的，考虑自己设置高斯噪声
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        # train数据使用Augmentation，val和test不使用
        self.transforms_train = T.Compose([
            # resize imgs to 224*224
            T.Resize((64, 64)),
            # T.CenterCrop(224),
            # 随机水平翻转
            T.RandomHorizontalFlip(0.5),
            # 随机垂直翻转
            T.RandomVerticalFlip(0.5),
            # 随机旋转正负90度
            T.RandomRotation([90, 270]),
            # 随机改变亮度、对比度
            T.ColorJitter(brightness=0.5, contrast=0.5),
            T.ToTensor(),
            normalize
        ])

        self.transforms_predict = T.Compose([
            T.Resize((64, 64)),
            # T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

    def load(self, path):
        self.backbone.load_state_dict(torch.load(
            path, map_location=torch.device('cuda')
            if self.do_use_cuda else torch.device('cpu')))

    def save(self, path):
        print("model are saved in:{}".format(path))
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(self.backbone.state_dict(),
                   time.strftime(path + self.name + '-' + '%Y-%m-%d.pth'))

    def predict_single(self, img):
        self.backbone.eval()
        if isinstance(img, np.ndarray):
            img = self.transforms_predict(Image.fromarray(img)).unsqueeze(0)
        return self.backbone(img.cuda() if self.do_use_cuda else img)[0].detach().tolist()

    def predict(self, img_list):
        self.backbone.eval()
        for i in range(len(img_list)):
            if isinstance(img_list[i], np.ndarray):
                img_list[i] = self.transforms_predict(Image.fromarray(img_list[i])).unsqueeze(0)
        return self.backbone(torch.cat(img_list).cuda()
                             if self.do_use_cuda else torch.cat(img_list)).detach().tolist()
