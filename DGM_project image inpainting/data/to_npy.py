import glob
import os
import cv2
import numpy as np

# first transform all pics to train to .npy
# do not need to load pics every time
# 0.95 data for training and 0.05 for validation
ratio = 0.95
# reduce input size in order to get result faster
image_size = 128

x = []
#paths = glob.glob('./images/*')
paths = glob.glob('E:/eclipse-workspace/data/img_align_celeba/*.jpg')
for path in paths[:500]:
    img = cv2.imread(path)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x.append(img)

x = np.array(x, dtype=np.uint8)
np.random.shuffle(x)

p = int(ratio * len(x))
x_train = x[:p]
x_test = x[p:]

if not os.path.exists('./npy'):
    os.mkdir('./npy')
np.save('./npy/x_train.npy', x_train)
np.save('./npy/x_test.npy', x_test)

