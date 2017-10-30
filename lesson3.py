from src.utils import *

#path = "data/dogscats/sample/"
path = "data/dogscats/"
model_path = path + 'models/'

if not os.path.exists(model_path):
    os.mkdir(model_path)

batch_size=64

model = vgg_ft(2)

model.load_weights(model_path+'finetune3.h5')

layers = model.layers

last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Convolution2D][-1]

lasy_conv_layer = layers[last_conv_idx]

conv_layers = layers[:last_conv_idx+1]
conv_model = Sequential(conv_layers)
fc_layers = layers[last_conv_idx+1:]

batches = get_batches(path+'train', shuffle=False, batch_size=batch_size)
val_batches = get_batches(path+'valid', shuffle=False, batch_size=batch_size)

trn_classes = batches.classes
val_classes = val_batches.classes

trn_labels = onehot(trn_classes)
val_labels = onehot(val_classes)

# predict conv
trn_features = conv_model.predict_generator(batches, batches.nb_sample)
val_features = conv_model.predict_generator(val_batches, val_batches.nb_sample)

save_array(model_path+'train_convlayer_features.bc', trn_features)
save_array(model_path+'vaild_convlayer_features.bc', val_features)

trn_features = load_array(model_path+'train_convlayer_features.bc')
val_features = load_array(model_path+'vaild_convlayer_features.bc')

print(trn_features.shape)



def proc_wghts(layer):
    return [w/2 for w in layer.get_weights()]

opt = RMSprop(lr=0.00001, rho=0.7)


def get_fc_model():
    model = Sequential([
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.0),
        Dense(4096,activation='relu'),
        Dropout(0.0),
        Dense(2, activation='softmax')
    ])

    for l1,l2 in zip(model.layers,fc_layers):
        l1.set_weights(l2.get_weights())

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

fc_model = get_fc_model()
fc_model.fit(trn_features,trn_labels, nb_epoch=8, batch_size=batch_size, validation_data=(val_features, val_labels))

fc_model.save_weights(model_path+'no_dropput.h5')
fc_model.load_weights(model_path+'no_dropput.h5')


print('================ ImageDataGenerator ===================')
# ImageDataGenerator
gen = image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

batches = get_batches(path+'train', gen, batch_size=batch_size)
val_batches = get_batches(path+'valid', shuffle=False, batch_size=batch_size)

fc_model = get_fc_model()

for layer in conv_model.layers:layer.trainable=False

conv_model.add(fc_model)

conv_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

conv_model.fit_generator(batches,samples_per_epoch=batches.nb_sample, nb_epoch=4, validation_data=val_batches,nb_val_samples=val_batches.nb_sample)

conv_model .save_weights(model_path+'aug1.h5')
conv_model .load_weights(model_path+'aug1.h5')


print('================ Batch normalization ===================')
#Batch normalization
def get_bn_layers(p):
    bn_layers = [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(p),

        BatchNormalization(),
        Dense(4096,activation='relu'),
        Dropout(p),
        BatchNormalization(),

        Dense(1000, activation='softmax')
    ]

    return bn_layers

p = 0.6
bn_model = Sequential(get_bn_layers(p))

bn_model.pop()
for layer in bn_model.layers:
    layer.trainable=True

bn_model.add(Dense(2, activation='softmax'))

bn_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

bn_model.fit(trn_features, trn_labels, nb_epoch=10, validation_data=(val_features, val_labels))

bn_model.save_weights(model_path+'bn.h5')
bn_model.load_weights(model_path+'bn.h5')

bn_layers = get_bn_layers(p)
bn_layers.pop()

bn_layers.append(Dense(2,activation='softmax'))

final_model = Sequential(conv_layers)
for layer in final_model.layers:
    layer.trainable = False
for layer in bn_layers:
    final_model.add(layer)

final_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
final_model.fit_generator(batches,samples_per_epoch=batches.nb_sample, nb_epoch=4, validation_data=val_batches,nb_val_samples=val_batches.nb_sample)

final_model.save_weights(model_path+'final1.h5')
final_model.load_weights(model_path+'final1.h5')