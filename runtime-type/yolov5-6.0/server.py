import time
from flask import Flask, jsonify, request
from yolov5 import YOLOv5
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)
det = YOLOv5()

def apply_clahe(image):
    """
    对输入图像应用 CLAHE 数据增强
    :param image: BGR 格式的 numpy 数组
    :return: 增强后的 BGR 图像
    """
    # 1. 将 BGR 转换为 LAB 格式（L代表亮度，A/B代表色彩）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. 创建 CLAHE 对象
    # clipLimit: 对比度限制阈值，通常设为 2.0-4.0
    # tileGridSize: 局部均衡化的网格大小，通常为 (8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # 3. 对 L 通道应用 CLAHE
    cl = clahe.apply(l)

    # 4. 合并通道并转回 BGR
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def json_result(val, names):
    keys = ['x0','y0','x1','y1','confidence','classId','area','className']
    objectVec = []
    for value_list in val:
        value_list.append(names[int(value_list[5])])
        classVec = {'classVec': {item[0]: item[1] for item in zip(keys, value_list)}}
        objectVec.append(classVec)
    result = {'objectVec':objectVec}
    return result

@app.route("/infer", methods=["POST"])
def infer():
    if request.method == "POST":
        try:
            res = json.loads(request.data)
            if not isinstance(res, dict):
                res = json.loads(res)
            input_image = res["image"]
            if input_image.startswith('b'):
                input_image = eval(input_image)
            input_image = base64.b64decode(input_image)
            imBytes = np.frombuffer(input_image, np.uint8)
            iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)

            # --- 数据增强：CLAHE ---
            iImage = apply_clahe(iImage)
            # ----------------------

            outs, names, tmp_weights = det.infer(iImage)
            
            print("time:===>", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            if (all(i is None for i in outs)) or (len(outs) == 0):
                result = {"success": False,"msg":'No detection results!'}
            else:
                result = json_result(outs.tolist(), names)
                result['model_path'] = tmp_weights
                result['success'] = True
                result['msg'] = 'Detection target output'
        except Exception as e:
            result = {'success': False,"msg":'Exception:'+str(e)}
        print(result)
    return jsonify(result)

if __name__ == "__main__":
    print("start.....................")
    app.run(host='0.0.0.0', debug=False, port=3334)