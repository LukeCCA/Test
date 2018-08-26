import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as Data
import lmdb

from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import sys

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import six
import sys
from PIL import Image

sys.path.insert(1,'./Test/')

import models.crnn as crnn
import dataset
import utils

batchSize = 64

outputPath ='/image/.Eric/lmdb_train'

train_dataset = dataset.lmdbDataset(root=outputPath)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize,shuffle=True,num_workers=4,
                                           collate_fn=dataset.alignCollate(imgH=32, imgW=320))

alphabet = '零壹貳參肆伍陸柒捌玖拾佰仟萬億兆元整'

nclass = len(alphabet) + 1
nc = 1
criterion = CTCLoss()

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()

model = crnn.CRNN(32, 1, 19, 256)
model = model.cuda()
model.load_state_dict(torch.load("/image/.Eric/modelResult/crnnV2.pkl"))

image = torch.FloatTensor(batchSize, 1, 32, 32)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

model = torch.nn.DataParallel(model)
image = image.cuda()
criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

optimizer = optim.Adam(model.parameters(), lr=0.01126,betas=(0.5, 0.999))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = model(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    model.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(20000):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        cost = trainBatch(model, criterion, optimizer)
        loss_avg.add(cost)
        
        i += 1

        if i % 10 == 0:
            print('[iteration%d/%d][%d/%d] Loss: %f' %
                  (epoch, 20000, i, len(train_loader), loss_avg.val() ))
            loss_avg.reset()
            
        # do checkpointing
        if i % 100 == 0:
            torch.save(model.module.state_dict(), '/image/.Eric/modelResult/netCRNN_{0}.pkl'.format(epoch))
            
torch.save(model.module.state_dict(), '/image/.Eric/modelResult/crnnV3.pkl')
