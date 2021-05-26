#!/usr/bin/env python
# coding: utf-8

# # $Load$ $and$ $prepare$ $dataset$

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers


# In[2]:


(images,labels),(_,_) = tf.keras.datasets.mnist.load_data()


# In[3]:


image_dim = images.shape
label_dim = labels.shape


# In[4]:


print("Image Data Dimension : {0} \nLabel Data Dimension : {1}".format(image_dim,label_dim))


# In[5]:


plt.imshow(images[0],cmap = 'gray')


# In[6]:


labels[0]


# In[7]:


# Normalize image between [-1 to 1]
images = (images - 127.5) / 127.5
images = images.reshape((image_dim[0],image_dim[1],image_dim[2],1)).astype('float32')
images.shape


# In[8]:


SHUFFLE_SIZE = 60000
BATCH_SIZE = 256
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
dataset.element_spec


# # $Define$ $generator$ $and$ $discriminator$

# In[9]:


def build_discriminator():
    model = tf.keras.Sequential(name = "Discriminator")
    # 1st Conv Layer
    model.add(layers.Conv2D(32, (5,5), strides = (1,1), padding = 'same', input_shape = [28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # 2nd Conv Layer
    model.add(layers.Conv2D(64, (5,5), strides = (2,2), padding = 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # 3rd Conv Layer
    model.add(layers.Conv2D(128, (5,5), strides = (2,2), padding = 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # Flatten & Output
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

discriminator = build_discriminator();
discriminator.summary()


# In[10]:


NOISE_SIZE = 100

def build_generator():
    model = tf.keras.Sequential(name = "Generator")
    model.add(layers.Dense(7*7*128,input_shape = (NOISE_SIZE,)))
    model.add(layers.Reshape((7,7,128)))
    # 1st Transposed Conv Layer
    model.add(layers.Conv2DTranspose(64, (5,5), padding = 'same', strides = (1,1), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # 2nd Transposed Conv Layer
    model.add(layers.Conv2DTranspose(32, (5,5), padding = 'same', strides = (2,2), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # 3rd Transposed Conv Layer
    model.add(layers.Conv2DTranspose( 1, (5,5), padding = 'same', strides = (2,2), use_bias=False))
    return model

generator = build_generator()
generator.summary()


# In[11]:


img = generator(tf.random.normal((10,NOISE_SIZE)))
plt.imshow(img[7],cmap = 'gray')


# In[12]:


discriminator(img)


# # $Set$ $training$ $loop$

# In[13]:


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# In[14]:


def dicriminator_loss(realOut,fakeOut):
    real_loss = bce(tf.ones_like(realOut),realOut)
    fake_loss = bce(tf.zeros_like(fakeOut),fakeOut)
    return real_loss+fake_loss

def generator_loss(fakeOut):
    return bce(tf.ones_like(fakeOut),fakeOut)


# In[15]:


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE,NOISE_SIZE])
    
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        # Generate Image from Noise
        fakeimg = generator(noise,training = True)
        # Get Prediction Output from Discriminator
        fakeOut = discriminator(fakeimg,training = True)
        realOut = discriminator(images,training = True)
        # Calculate Loss
        disc_loss = dicriminator_loss(realOut,fakeOut)
        gen_loss  = generator_loss(fakeOut)
    
    # Calculate Gradients
    discriminator_gradient = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    generator_gradient = gen_tape.gradient(gen_loss,generator.trainable_variables)
    
    # Apply Gradients
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient,discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(generator_gradient,generator.trainable_variables))


# In[16]:


import time
num_sample_img = 16

def train(dataset,epochs):
    start = time.time()
    for epoch in range(epochs):
        for img_batch in dataset:
            train_step(img_batch)
        print ("Time for epoch {0} is {1} sec".format(epoch + 1, time.time()-start))
        
        img = generator(tf.random.normal((num_sample_img,NOISE_SIZE)),training=False)
        fig = plt.figure(figsize = (4,4))
        for i in range(img.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(img[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig('image_at_epoch_{:02d}.png'.format(epoch))
        plt.show()


# # $Train$ $GAN$

# In[17]:


EPOCHS = 50
train(dataset, EPOCHS)


# In[18]:


generator.save('Generator')
discriminator.save('Discriminator')


# In[ ]:




