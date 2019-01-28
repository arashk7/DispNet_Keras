from DispNet_AK import DispNet as dsn
from PIL import Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt
_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def load_input_dataset(path) -> object:
    files = os.listdir(path)
    # x = np.zeros((len(files), int(100), int(300)))
    x = np.zeros((len(files), int(128), int(512)))
    for i in range(len(files)):
        # img = load_image(path + "/" + files[i])
        img = Image.open(path + "/" + files[i]).convert('L')
        img.load()
        img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
        data = np.asarray(img, dtype="int32")
        x[i, :, :] = data

    return x


def load_output_dataset(path) -> object:
    files = os.listdir(path)
    # x = np.zeros((len(files), int(100), int(150)))
    x = np.zeros((len(files), int(64), int(128)))
    for i in range(len(files)):
        img = Image.open(path + "/" + files[i]).convert('L')
        img.load()
        img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
        data = np.asarray(img, dtype="int32")
        x[i, :, :] = data

    return x


# Initialize Dataset
path = "../../Dataset/DepthMap_dataset-master/Stereo"
x = load_input_dataset(path)
X = np.zeros((x.shape[0], x.shape[1], int(x.shape[2] / 2), 2))
X[:, :, :, 0] = x[:, :x.shape[1], :int(x.shape[2] / 2)]
X[:, :, :, 1] = x[:, :x.shape[1], int(x.shape[2] / 2):]
min = np.min(np.min(X))
max = np.max(np.max(X))
X = (X - min) / max

path = "../../Dataset/DepthMap_dataset-master/Depth_map"
y = load_output_dataset(path)
Y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 1))
Y[:, :, :, 0] = y[:, :, :]
min = np.min(np.min(Y))
max = np.max(np.max(Y))
Y = (Y - min) / max

# Initialize the SCN network
model_name = 'model/model_DispNet_1'

model = dsn.load_model_and_weight(model_name)
model = dsn.compile(model)

p = model.predict(X, batch_size=dsn.get_batch_size(), verbose=0)
img = p[1, :, :, 0]
# img = img * 255
plt.imshow(img,cmap="rainbow")
plt.show()