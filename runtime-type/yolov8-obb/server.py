import time
import yaml
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from flask import Flask, jsonify, request
import cv2
import base64
import json

with open("./weights/model.yaml", 'r') as file:
    config = yaml.safe_load(file)
    weights = config['weights']

det= YOLO(weights)
app = Flask(__name__)
def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)
    ctr = center[..., :2]
    w, h, angle = (center[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 =  torch.cat(vec1, dim=-1)
    vec2 =  torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return  torch.stack([pt1, pt2, pt3, pt4], dim=-2)

def json_result(val,confs ):
    keys = ['point','confidence', 'classId', 'className']
    objectVec = []
    for i,value_item in enumerate(val):
        value_list =[]
        for j,point in enumerate(value_item):
            value_item[j-1]=max(point,0) 
        value_list.append(value_item)
        #print("value_list:==>",value_list)
        value_list.append(confs[i])
        value_list.append(0)
        value_list.append(str('jjd'))
        #print(value_list)
        classVec = {'classVec': {item[0]: item[1] for item in zip(keys, value_list)}}
        objectVec.append(classVec)
    result = {'objectVec': objectVec}
    return result

@app.route("/infer", methods=["POST"])
def infer():
    if request.method == "POST":
        try:
            res = json.loads(request.data) # usetiem: 0.04921922698849812
            if not isinstance(res, dict):
                res = json.loads(res)
            input_image = res["image"]
            # 如果是字节流的字符串直接执行
            if input_image.startswith('b'):
                input_image = eval(input_image)
            input_image = base64.b64decode(input_image) #usetiem: 0.00177
            imBytes = np.frombuffer(input_image, np.uint8)
            iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR) # usetiem: 0.020s
            results = det(iImage)[0]   #usetiem: 0.015s
            boxes = results.obb.data.cpu()
            # boxes = results.obb.data
            
            if (all(i is None for i in boxes)) or (len(boxes) == 0):
                result = {"success": False, "msg": 'No detection results!'}
            else:
                confs = boxes[..., 5].tolist()
                boxes = xywhr2xyxyxyxy(boxes[..., :5]).view(len(boxes),-1)
                # print(boxes.tolist())
                result = json_result(boxes.tolist(), confs)
                # print(result)
                result['model_path'] = weights
                result['success'] = True
                result['msg'] = 'Detection target output'
                # print(result)
        except Exception as e:
            result = {'success': False, "msg": 'Exception:' + str(e)}
            pass
    return jsonify(result)


if __name__ == "__main__":
    print("start.....................")
    app.run(host='0.0.0.0', debug=False, port=3333)
