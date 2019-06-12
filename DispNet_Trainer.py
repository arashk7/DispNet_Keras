from DispNet_AK.DispNet import DispNet
from PIL import Image
import numpy as np
import os
import time

_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


# Load the input stereo images from dataset directory
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


# Load the output disparity map from dataset directory
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


# Initialize Dataset (network input)
path = "../../Dataset/DepthMap_dataset-master/Stereo"
x = load_input_dataset(path)
X = np.zeros((x.shape[0], x.shape[1], int(x.shape[2] / 2), 2))
X[:, :, :, 0] = x[:, :x.shape[1], :int(x.shape[2] / 2)]
X[:, :, :, 1] = x[:, :x.shape[1], int(x.shape[2] / 2):]
min = np.min(np.min(X))
max = np.max(np.max(X))
X = (X - min) / max

# Initialize Dataset (network output)
path = "../../Dataset/DepthMap_dataset-master/Depth_map"
y = load_output_dataset(path)
Y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 1))
Y[:, :, :, 0] = y[:, :, :]
min = np.min(np.min(Y))
max = np.max(np.max(Y))
Y = (Y - min) / max

# Initialize the DispNet network
model_name = 'model_DispNet_1'
first_run = True
dsn = DispNet()
if first_run:
    # create model
    dsn.init_model()
    dsn.compile()
else:
    # load model
    dsn.load_model_and_weight('model/' + model_name)
    dsn.compile()

min_err = 1000
n_epoch_save = 100
counter = 0
epoch_co = 0
tic()
# Train loop
for i in range(epoch_co, 3000):
    # Train the model for one epoch
    dsn.train(X, Y, x_vl, y_vl, n_epoch=1, batch_size=10)
    # Check the error
    err = dsn.get_error_rate(X, Y)

    # If the network improved then save the network weights
    if err < min_err:
        dsn.save_model_and_weight('model/' + model_name)
        min_err = err
    else:
        print("epoch: " + str(i) + "   error:" + str(err))

    if counter >= n_epoch_save:
        dsn.save_model_and_weight('model/' + model_name + "_epoch_" + str(i))
        counter = 0
    counter += 1

tac()
