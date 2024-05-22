#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

CLASSES = ['Pedestrian', 'Car', 'Cyclist']

class_num = len(CLASSES)
input_h = 384
input_w = 1280

object_thresh = 0.5

output_h = 96
output_w = 320

downsample_ratio = 4


class ScoreXY:
    def __init__(self, score, c, h, w):
        self.score = score
        self.c = c
        self.h = h
        self.w = w


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image /= 255.0
    image -= mean
    image /= std

    image = image.transpose((2, 0, 1))

    return image


def nms(heatmap, heatmapmax):
    keep_heatmap = []
    for b in range(1):
        for c in range(class_num):
            for h in range(output_h):
                for w in range(output_w):
                    if heatmapmax[c * output_h * output_w + h * output_w + w] == heatmap[c * output_h * output_w + h * output_w + w] and heatmap[c * output_h * output_w + h * output_w + w] > object_thresh:
                        temp = ScoreXY(heatmap[c * output_h * output_w + h * output_w + w], c, h, w)
                        keep_heatmap.append(temp)
    return keep_heatmap


def postprocess(outputs):
    heatmap = outputs[0]
    offset_2d = outputs[1]
    size_2d = outputs[2]
    heatmapmax = outputs[3]

    keep_heatmap = nms(heatmap, heatmapmax)
    top_heatmap = sorted(keep_heatmap, key=lambda t: t.score, reverse=True)

    boxes2d = []
    for i in range(len(top_heatmap)):
        if i > 50:
            break

        classId = top_heatmap[i].c
        score = top_heatmap[i].score
        w = top_heatmap[i].w
        h = top_heatmap[i].h

        bx = (w + offset_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        by = (h + offset_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bw = (size_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bh = (size_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio

        xmin = (bx - bw / 2) / input_w
        ymin = (by - bh / 2) / input_h
        xmax = (bx + bw / 2) / input_w
        ymax = (by + bh / 2) / input_h

        keep_flag = 0
        for j in range(len(boxes2d)):
            xmin1 = boxes2d[j].xmin
            ymin1 = boxes2d[j].ymin
            xmax1 = boxes2d[j].xmax
            ymax1 = boxes2d[j].ymax
            if IOU(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1) > 0.45:
                keep_flag += 1
                break
        if keep_flag == 0:
            bbox = DetectBox(classId, score, xmin, ymin, xmax, ymax)
            boxes2d.append(bbox)

    return boxes2d


def detect(img_path):
    origin_image = cv2.imread(img_path)
    image_h, image_w = origin_image.shape[0:2]
    resize_image = cv2.resize(origin_image, (input_w, input_h))
    image = precess_image(resize_image, input_w, input_h)

    image = np.expand_dims(image, axis=0)

    ort_session = ort.InferenceSession('./centernet.onnx')
    ort_outputs = (ort_session.run(None, {'data': image}))

    outputs = []
    for i in range(len(ort_outputs)):
        outputs.append(ort_outputs[i].reshape(-1))

    result = postprocess(outputs)

    print('detect num is:', len(result))

    for i in range(len(result)):
        classid = result[i].classId
        score = result[i].score
        xmin = int(result[i].xmin * image_w + 0.5)
        ymin = int(result[i].ymin * image_h + 0.5)
        xmax = int(result[i].xmax * image_w + 0.5)
        ymax = int(result[i].ymax * image_h + 0.5)

        cv2.rectangle(origin_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = '%s:%.2f' % (CLASSES[classid], score)
        cv2.putText(origin_image, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_onnx_result.jpg', origin_image)


if __name__ == '__main__':
    print('This is main ....')
    img_path = './test.png'
    detect(img_path)
