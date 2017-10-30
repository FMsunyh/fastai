import os

import bcolz
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from utils import get_data

from src.vgg16 import Vgg16

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# def limit_men():
#     cfg = K.tf.ConfigProto()
#     cfg.gpu_options.allow_growth = True
#     K.set_session(K.tf.Session(config=cfg))
#
# limit_men()
vgg = Vgg16()
model = vgg.model

path = "data/dogscats/sample/"
# path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

# batch_size = 4
batch_size = 100


batches = vgg.get_batches(path+'train', shuffle=False, batch_size=1)
val_batches = vgg.get_batches(path+'valid', shuffle=False, batch_size=1)
# test_batches = vgg.get_batches(path+'test', shuffle=False, batch_size=1)

def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
def load_array(fname):
    return bcolz.open(fname)[:]

trn_data = get_data(path+'train')

val_data = get_data(path+'valid')


# save_array(model_path+'train_data.bc', trn_data)
# save_array(model_path+'valid_data.bc', val_data)
#
# trn_data = load_array(model_path+'train_data.bc')
# val_data = load_array(model_path+'valid_data.bc')

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)


trn_features = model.predict(trn_data, batch_size=batch_size)
val_features = model.predict(val_data, batch_size=batch_size)


# save_array(model_path+'train_lastlayer_features.bc', trn_features)
# save_array(model_path+'valid_lastlayer_features.bc', val_features)
#
#
# trn_features = load_array(model_path+'train_lastlayer_features.bc')
# val_features = load_array(model_path+'valid_lastlayer_features.bc')

lm = Sequential([ Dense(2, activation='softmax', input_shape = (1000,)) ])
lm.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

lm.fit(trn_features, trn_labels, nb_epoch=3, batch_size=batch_size, validation_data=(val_features, val_labels))

lm.summary()

model.summary()

model.pop()
for layer in model.layers:
    layer.trainable=False

model.summary()

model.add(Dense(2, activation='softmax'))

model.summary()

gen = image.ImageDataGenerator()
batches = gen.flow(trn_data, trn_labels, batch_size=batch_size, shuffle=True)
val_batches = gen.flow(val_data, val_labels, batch_size=batch_size, shuffle=True)

opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

def fit_model(model, batches, val_batches, nb_epoch=1):
    model.fit_generator(batches, samples_per_epoch=batches.n, nb_epoch=nb_epoch, validation_data=val_batches,nb_val_samples=val_batches.n)

fit_model(model, batches, val_batches, nb_epoch=4)


model.save_weights(model_path+'finetune1.h5')
model.load_weights(model_path+'finetune1.h5')

model.evaluate(val_data,val_labels)

preds = model.predict_classes(val_data, batch_size=batch_size)
probs = model.predict_proba(val_data, batch_size=batch_size)[:,0]