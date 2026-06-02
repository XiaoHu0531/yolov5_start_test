import cv2
import torch
import numpy as np
from ultralytics import YOLO
from flask import Flask, jsonify, request
import cv2
import base64
import json

model = YOLO("weights/best.pt")
app = Flask(__name__)


def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    print("is_numpy: ",is_numpy)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)


def json_result(val):
    keys = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'confidence', 'classId', 'className']
    objectVec = []
    for value_list in val:
        ###value_list
        value_list.append(str('jjd'))
        classVec = {'classVec': {item[0]: item[1] for item in zip(keys, value_list)}}
        objectVec.append(classVec)
    result = {'objectVec': objectVec}
    return result


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

# if __name__ == "__main__":
#     print("start.....................")
#     # app.run(host='0.0.0.0', debug=False, port=5001)

if __name__ == "__main__":

    model = YOLO("/home/ps/documents/hx/code项目汇总/yolov8/runs/obb/train7/weights/best.pt")
    img = cv2.imread("/home/ps/documents/yolov5_obb/runs/detect/exp2/微信图片_20221209092035.jpg")
    results = model(img)[0]
    names = results.names
    boxes = results.obb.data.cpu()
    print(boxes)
    confs = boxes[..., 5].tolist()
    print("confs======>",confs)
    classes = list(map(int, boxes[..., 6].tolist()))
    print("classes======>",classes)
    boxes = xywhr2xyxyxyxy(boxes[..., :5]).view(2,-1)

    print("boxes======>",len(boxes))

    # for i, box in enumerate(boxes):
    #     confidence = confs[i]
    #     label = classes[i]
    #     color = random_color(label)
    #     cv2.polylines(img, [np.asarray(box, dtype=int)], True, color, 2)
    #     caption = f"{names[label]} {confidence:.2f}"
    #     print(caption)
    #     w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
    #     left, top = [int(b) for b in box[0]]
    #     cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
    #     cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
    #
    # cv2.imwrite("predict-obb.jpg", img)
    # print("save done")
