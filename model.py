import torch
from torch import nn
from torchvision import models
import numpy as np
import cv2

class AbandonDetector(nn.module):
    def __init__():
        super(AbandonDetector, self).__init__()
        self.inception = models.inception_v3(pretrained=True)

inception = models.inception_v3(pretrained=True)
inception = inception.cuda()
frames = np.load('data/tensor0.npy')

batch = []
for i in range(10):
    frame = cv2.resize(frames[i], (299, 299), cv2.INTER_AREA)
    batch.append(frame)
batch = np.stack(batch)
batch = torch.FloatTensor(batch).transpose(1, 3).cuda()

out = inception(batch)
print(out)