# from theano.sandbox import cuda
# cuda.use('gpu1')

from utils import *


#path = "data/StateFarm/"
path = "data/StateFarm/sample/"

batch_size=64

batches = get_batches(path+'train', batch_size=batch_size)
val_batches = get_batches(path+'valid', batch_size=batch_size)

model = Sequential( [
    BatchNormalization(axis=1,input_shape=(3,224,224)),
    Flatten(),
    Dense(10,activation='softmax')
     ])



# model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

# change the learning rate
model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


model.optimizer.lr = 0.0001
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, nb_val_samples=val_batches.nb_sample)
# model.summary()

