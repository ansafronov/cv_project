#!/usr/bin/env python
# coding: utf-8

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# model = YOLO("yolov8m.pt")
# torch.save(model, "yolov8m.pt")
model = YOLO('yolov8x-seg.pt')

def detect_bottle(imgpath, return_mask=True):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bottle_cls = [39., 67.]
    results = model.predict(imgpath)
    indices = np.where(np.array(results[0].boxes.cls == bottle_cls[0]))[0]
    if len(indices) == 0:
        index = np.where(np.array(results[0].boxes.cls == bottle_cls[1]))[0][0]
    else:
        index = indices[0]
    x1, y1, x2, y2 = results[0].boxes.xyxy[index]
    yy1, yy2 = max(0, y1 - 0.1 * (y2-y1)), min(img.shape[0], y2 + 0.1 * (y2-y1))
    xx1, xx2 = max(0, x1 - 0.1 * (x2-x1)), min(img.shape[1], x2 + 0.1 * (x2-x1))
    imgcenter = img[int(yy1):int(yy2), int(xx1):int(xx2)]
    
    if return_mask:
        mask = (results[0].masks.data[index].numpy() * 255).astype("uint8")
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        maskcenter = mask[int(yy1):int(yy2), int(xx1):int(xx2)]
        return imgcenter, maskcenter
    
    return imgcenter

def get_6_points(img, mask):
    nonzero_y = np.where(np.sum(mask, axis=1) > 0)[0]
    yhigh, ylow = nonzero_y[0], nonzero_y[-1]
    nonzero_x = np.where(mask[yhigh] > 0)[0]
    B = (int(np.mean(nonzero_x)), yhigh)
    nonzero_x = np.where(mask[ylow] > 0)[0]
    E = (int(np.mean(nonzero_x)), ylow)

    tg = 2
    coordinates = [0]*4
    for i, point in enumerate(['A', 'C', 'D', 'F']):
        for y in range(img.shape[0]):
            if point == 'A':
                x1, y1 = 0, y
                x2, y2 = int(y / tg), 0
            elif point == 'C':
                x1, y1 = img.shape[1], y
                x2, y2 = img.shape[1] - int(y / tg), 0
            elif point == 'F':
                x1, y1 = 0, img.shape[0] - y
                x2, y2 = int(y / tg), img.shape[0]
            else:
                x1, y1 = img.shape[1], img.shape[0] - y
                x2, y2 = img.shape[1] - int(y / tg), img.shape[0]
            imgzero = np.zeros_like(img)
            cv2.line(imgzero, (x1, y1), (x2, y2), 255, 10)
            imgzero = cv2.cvtColor(imgzero, cv2.COLOR_BGR2GRAY)
            if (imgzero * mask ).sum() > 0:
                break
        ys, xs = np.where((imgzero * mask ) > 0)
        coordinates[i] = xs[-1], ys[-1]
    A, C, D, F = coordinates
    return A, B, C, D, E, F

def mouseCB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print (x*2, y*2)
        points.append([x*2, y*2])

# Double-click to mark a point
# Add clockwise starting from the top left to the bottom left (from A to F)
# Press 'q' at the end
def get_6_points_manually(img):
    global points
    points = []
    exit_flag = True

    window_name = 'image'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseCB)
    imgres = cv2.resize(img, (int(img.shape[1] // 2), int(img.shape[0] // 2)))
    cv2.imshow(window_name, imgres)

    while True:

        ip = cv2.waitKey(0) & 0xFF

        if ip == ord('q'):
            exit_flag = False
            cv2.destroyAllWindows()
            break
    return points