import os
import random

# trainval_percent = 0.2  # 可自行进行调节
train_percent = 0.9
xmlfilepath = '/home/ps/model/yolov5/data/fire/Annotations'
txtsavepath = '/home/ps/model/yolov5/data/fire/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
# tv = int(num * trainval_percent)
tr = int(num * train_percent)
# trainval = random.sample(list, tv)
train = random.sample(list, tr)

# ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('/home/ps/model/yolov5/data/fire/ImageSets/Main/test.txt', 'w')
ftrain = open('/home/ps/model/yolov5/data/fire/ImageSets/Main/train.txt', 'w')
# fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in train:
        ftrain.write(name)
    else:
        ftest.write(name)
# for i in list:
#     name = total_xml[i][:-4] + '\n'
#     if i in trainval:
#         # ftrainval.write(name)
#         if i in train:
#             ftest.write(name)
#         # else:
#         # fval.write(name)
#     else:
#         ftrain.write(name)

# ftrainval.close()
ftrain.close()
# fval.close()
ftest.close()
