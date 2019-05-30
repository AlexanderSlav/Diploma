'''
The squeezenet classifies whether the image is a close-up or not.
The next step is send the image to ssd input trained on the dataset with the close-up or not close-up images
depending on the output of the previous net.
'''
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision import datasets, models, transforms
from ssd import build_ssd
from data  import VOC_CLASSES as labels


def data_preparation(input_size,frame):
    x = cv2.resize(frame, (input_size,input_size)).astype(np.float32)

    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    return xx


def main():

    path = "C:/diploma/ssd-pytorch"
    path_to_weights = "C:/diploma/ssd-pytorch/weights"
    input_stream = path + "/" + "krasnodar_zenit.mp4"

    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    model_ft = models.squeezenet1_0(True)
    model_ft.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = 2

    model_ft.load_state_dict(torch.load(path + '/' + 'bigsmall_mymodel.pth'))
    labels_small_big = ['small','big']
    labels_my_dataset = [('player')]



    while cap.isOpened():
        ret, frame = cap.read()
        #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        initial_w = cap.get(3)
        initial_h = cap.get(4)

        xx = data_preparation(224, frame)
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = model_ft(xx)

        if y[0][0].item() > y[0][1].item():
            cv2.putText(frame, labels_small_big[1], (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            xx_big = data_preparation(300, frame)
            net = build_ssd('test', 300, 21)  # initialize SSD
            net.load_weights(path_to_weights + '/' + 'ssd300_mAP_77.43_v2.pth')
            if torch.cuda.is_available():
                xx_big = xx_big.cuda()
            y_big = net(xx_big)
            detections = y_big
            for i in range(detections.size(1)):
                j= 0
                while detections[0,i,j,0] >= 0.5:
                    score = detections[0,i,j,0]
                    label_name = labels[i-1]
                    class_id = i - 1
                    xmin = int(detections[0,i,j,1].item() * initial_w)
                    ymin = int(detections[0,i,j,2].item() * initial_h)
                    xmax = int(detections[0,i,j,3].item() * initial_w)
                    ymax = int(detections[0,i,j,4].item() * initial_h)
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, label_name + ' ' + str(round(score.item()* 100, 1)) + ' %', (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    j += 1
        else:
            cv2.putText(frame, labels_small_big[0], (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            xx_small = data_preparation(300, frame)
            net = build_ssd('test', 300, 2)  # initialize SSD
            net.load_weights(path_to_weights + '/' + 'weights_my_dataset.pth')
            if torch.cuda.is_available():
                xx_small = xx_small.cuda()
            y_small = net(xx_small)
            detections = y_small
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.2:
                    score = detections[0, i, j, 0]
                    label_name = labels_my_dataset[i-1]
                    class_id = i - 1
                    xmin = int(detections[0, i, j, 1].item() * initial_w)
                    ymin = int(detections[0, i, j, 2].item() * initial_h)
                    xmax = int(detections[0, i, j, 3].item() * initial_w)
                    ymax = int(detections[0, i, j, 4].item() * initial_h)
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, label_name + ' ' + str(round(score.item() * 100, 1)) + ' %',
                                (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    j += 1
        cv2.imshow("Detection Results", frame)
        key = cv2.waitKey(1)
        if key == 'q':
            break

cv2.destroyAllWindows()
if __name__ == '__main__':
    sys.exit(main())
