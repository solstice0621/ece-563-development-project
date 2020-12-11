#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf  
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics  
import os, glob
import random, csv
import matplotlib.pyplot as plt


# In[ ]:


# load csv and character array
def load_csv(root, filename, name2label):
    if not os.path.exists(os.path.join(root, filename)):  # if csv is not exist，establish a new one
        images = []  # initial 
        for name in name2label.keys():  # get path
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        print(len(images), images)  # print length and path
        random.shuffle(images)  
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  
                name = img.split(os.sep)[-2]  
                label = name2label[name]  # find label
                writer.writerow([img, label])  # load csv
            print('written into csv file:', filename)
    
    images, labels = [], [] 
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:  
            img, label = row  
            label = int(label)  # change char to int
            images.append(img)  
            labels.append(label)
    assert len(images) == len(labels)  
    return images, labels


# In[ ]:


# divide dataset
def load_pokemon(root, mode='train'):
    name2label = {}  # create{key:value} to save labels
    for name in sorted(os.listdir(os.path.join(root))):  
        if not os.path.isdir(os.path.join(root, name)): 
            continue
        name2label[name] = len(name2label.keys())   
    images, labels = load_csv(root, 'images.csv', name2label)  # load path and label
    if mode == 'train':  # 60%
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':  # 20% = 60%->80%
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:  # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return images, labels, name2label


# In[ ]:


# preprocess
def preprocess(image_path, label):
    x = tf.io.read_file(image_path)  # load images
    x = tf.image.decode_jpeg(x, channels=3)  # matrix
    x = tf.image.resize(x, [244, 244])
    x = tf.image.random_crop(x, [224, 224, 3])  # cut
    x = tf.cast(x, dtype=tf.float32) / 255.  # normalize
    x = normalize(x)
    y = tf.convert_to_tensor(label)  # tensor
    return x, y

# load dataset
images, labels, table = load_pokemon('pokemon', 'train')
print('images', len(images), images)
print('labels', len(labels), labels)
print(table)
db = tf.data.Dataset.from_tensor_slices((images, labels))  # create db
db = db.shuffle(1000).map(preprocess).batch(32).repeat(20)  # batch=32，train 20 times


# In[ ]:


# networks
network = Sequential([
    # first layer
    layers.Conv2D(48, kernel_size=11, strides=4, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),  # 55*55*48
    layers.MaxPooling2D(pool_size=3, strides=2),  # 27*27*48
    # second layer
    layers.Conv2D(128, kernel_size=5, strides=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),  # 27*27*128
    layers.MaxPooling2D(pool_size=3, strides=2),  # 13*13*128
    #  third layer
    layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),  # 13*13*192
    # fourth layer
    layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),  # 13*13*192
    # fifth layer
    layers.Conv2D(128, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),  # 13*13*128
    layers.MaxPooling2D(pool_size=3, strides=2),  # 6*6*128
    layers.Flatten(),  # 6*6*128=4608
    # 6th layer
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    # 7th layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.5),
    # 8th layer（output）
    layers.Dense(5)  
])
network.build(input_shape=(32, 224, 224, 3))  
network.summary()


# In[ ]:


# training
optimizer = optimizers.SGD(lr=0.01)  # Batch Gradient Descent Learning rate=0.01
acc_meter = metrics.Accuracy()
x_step = []
y_accuracy = []
for step, (x, y) in enumerate(db):  
    with tf.GradientTape() as tape:  
        x = tf.reshape(x, (-1, 224, 224, 3))  # input[b, 224, 224, 3]
        out = network(x)  # output[b, 5]
        y_onehot = tf.one_hot(y, depth=5)  # one-hot
        loss = tf.square(out - y_onehot)
        loss = tf.reduce_sum(loss)/32  # batch=32，Mean Square Error Loss
        grads = tape.gradient(loss, network.trainable_variables)  #  compute gradient
        optimizer.apply_gradients(zip(grads, network.trainable_variables))  # update parameters
        acc_meter.update_state(tf.argmax(out, axis=1), y)  # compare labels
    if step % 10 == 0:  # 每200个step，打印一次结果
        print('Step', step, ': Loss is: ', float(loss), ' Accuracy: ', acc_meter.result().numpy())
        x_step.append(step)
        y_accuracy.append(acc_meter.result().numpy())
        acc_meter.reset_states()

