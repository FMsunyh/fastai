import os
import shutil

import numpy as np
from six.moves import range

from src import image_gen

# from keras.preprocessing import image

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def remove_all(path):
    try:
        filelist = os.listdir(path)
        for f in filelist:
            filepath = os.path.join(path, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(filepath + " removed!")
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath, True)
                print("dir " + filepath + " removed!")
    except IOError as exc:
        print(exc)

#path = "data/image_generator/"
path = "data/image_generator/sample/"
if not os.path.exists(path):
    os.makedirs(path)

gen_path = path+ "gen/p/"
if not os.path.exists(gen_path):
    os.makedirs(gen_path)
else:
    remove_all(gen_path)

datagen = image_gen.ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25,
                                       fill_mode='nearest', channel_shift_range=100,
                                       rotation_range=30, zoom_range=[1.6, 1.6])

# order_list = list(np.random.permutation(10000))
order_list = list(np.arange(5000))

for j in range(5):
    gen_data = datagen.flow_from_directory(path+'train/',
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=gen_path,
                                       save_prefix='p',
                                       target_size=(224, 224))
    for i in range(1000):
        print(1000*j+ i)
        gen_data.next(order_list[ 1000*j+ i])
