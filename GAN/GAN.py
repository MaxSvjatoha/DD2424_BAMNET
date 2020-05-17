#import all the library
import os
import tensorflow as tf
import numpy as np
from glob import glob
import random
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import keras
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import , Conv2D , Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from PIL import Image


#Hyperparameters
im_size = 128       #image size
noise_shape= 100    #noise shape
lr_D = 0.0002       #learning rate fpr the discriminator
lr_G = 0.0002       #learning rate fpr the Generator
batch_size = 32     #BAtch size
EPOCHS = 1000       #number of epochs
beta = 0.5          #beta value for batch optimization with Adam
init_stddev= 0.02   #initialition of the weights
eps = 0.00005       #epsilon value avoid div
num_sample = 8      #number of sample output
moment = 0.9        #momentum
opt_G=Adam(lr=lr_G, beta_1=beta)        #Optimizer for the Generator
opt_D=Adam(lr=lr_G, beta_1=beta)        #Optimizer for the Discriminator

# Create the Generator
def Generator(z=(noise_shape,)):
    # 4 x 4 x 512
    input = Input(z)
    hid_layer = Dense(4 * 4 * 512, activation ='relu', name="Dense")(input)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)
    hid_layer = Reshape((4, 4, 512))(hid_layer)

    # 4x4x512 -> 8x8x512
    hid_layer = Conv2DTranspose(512, kernel_size = [5, 5],
                          strides = [2, 2],
                          padding="same",
                          kernel_initializer = keras.initializers.TruncatedNormal(stddev=init_stddev),
                          )(hid_layer)
    hid_layer = BatchNormalization(momentum = moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 8x8x512 -> 16x16x256
    hid_layer = Conv2DTranspose(256, kernel_size=[5, 5],
                          strides=[2, 2],
                          padding="same",
                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                          )(hid_layer)
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 16x16x256 -> 32x32x128
    hid_layer = Conv2DTranspose(128, kernel_size=[5, 5],
                          strides=[2, 2],
                          padding="same",
                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                          )(hid_layer)
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 32x32x128 -> 64x64x64
    hid_layer = Conv2DTranspose(64, kernel_size=[5, 5],
                          strides=[2, 2],
                          padding="same",
                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                          )
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 64x64x64 -> 128x128x32
    hid_layer = Conv2DTranspose(32, kernel_size=[5, 5],
                          strides=[2, 2],
                          padding="same",
                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                          )(hid_layer)
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 128x128x32 -> 128x128x3
    hid_layer = Conv2DTranspose(3, kernel_size=[5, 5],
                          strides=[1, 1],
                          padding="same",
                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                          )(hid_layer)

    out = Activation("tanh")(hid_layer)

    model = Model(inputs=input, outputs=out)
    model.summary()

    return model


# Create  the Discriminator
def Discriminator(input_shape=(im_size, im_size, 3)):
    # 64x64x3 -> 32x32x32
    input = Input(input_shape)
    hid_layer = Conv2D(filters=32,
                 kernel_size=[5, 5],
                 strides=[2, 2],
                 padding="same",
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                 )(input)
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 32x32x32-> 16x16x64
    hid_layer = Conv2D(filters=64,
                 kernel_size=[5, 5],
                 strides=[2, 2],
                 padding="same",
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                 )(hid_layer)
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 16x16x64  -> 8x8x128
    hid_layer = Conv2D(filters=128,
                 kernel_size=[5, 5],
                 strides=[2, 2],
                 padding="same",
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                 )(hid_layer)
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 8x8x128 -> 8x8x256
    hid_layer = Conv2D(filters=256,
                 kernel_size=[5, 5],
                 strides=[1, 1],
                 padding="same",
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                 )(hid_layer)
    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    # 8x8x256 -> 8x8x512
    hid_layer = Conv2D(filters=512,
                 kernel_size=[5, 5],
                 strides=[2, 2],
                 padding="same",
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                 )(hid_layer)

    hid_layer = BatchNormalization(momentum=moment, epsilon=eps)(hid_layer)
    hid_layer = LeakyReLU(alpha=0.2)(hid_layer)

    hid_layer = Flatten()(hid_layer)

    out = Dense(1, activation='sigmoid')(hid_layer)
    model = Model(inputs=input, outputs=out)

    model.summary()

    return model


#Discriminator
discriminator = Discriminator((im_size, im_size,3))
discriminator.compile(loss='binary_crossentropy',optimizer=opt_D,metrics=['accuracy'])

discriminator.trainable = False

#Generator
generator = Generator((noise_shape,))

#GAN
GAN_input = Input(shape=(noise_shape,))
x = generator(GAN_input)
GAN_output = discriminator(x)
GAN = Model(GAN_input, GAN_output)
GAN.summary()
GAN.compile(loss='binary_crossentropy',optimizer=opt_G)


#plot and print the loss
def summarize_epoch(D_loss, G_loss, data_shape, epoch, input_z):
    minibatch_size = int(data_shape[0] // batch_size)
    print("Epoch {}/{}".format(epoch, EPOCHS),
          "\nD Loss: {:.5f}".format(np.mean(D_loss[-minibatch_size:])),
          "\nG Loss: {:.5f}".format(np.mean(G_loss[-minibatch_size:])))
    fig, ax = plt.subplots()
    plt.plot(D_loss, label='Discriminator', alpha=0.6)
    plt.plot(G_loss, label='Generator', alpha=0.6)

    plt.title("Losses")
    plt.legend()
    plt.show()
    plt.close()
    test(input_z, epoch)


def get_batches(data):
    batches = []
    for i in range(int(data.shape[0] // batch_size)):
        batch = data[i *  batch_size:(i + 1) *  batch_size]
        augmented_images = []
        for img in batch:
            image = Image.fromarray(img)
            if random.choice([True, False]):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(np.asarray(image))
        batch = np.asarray(augmented_images)
        norm_batch = (batch / 127.5) - 1.0
        batches.append(norm_batch)
    return np.array(batches)


def test(input_z, epoch):
    samples = generator.predict(input_z[:num_sample])
    sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]
    show_samples(sample_images, out_put_data_dir + "samples", epoch)



#Show the sample after each epoch and save them into the folder
def show_samples(sample_images, name, epoch):
        figure, axes = plt.subplots(1, len(sample_images), figsize=(im_size, im_size))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = sample_images[index]
            axis.imshow(image_array)
            image = Image.fromarray(image_array)
            image.save(name + "_" + str(epoch) + "_" + str(index) + ".png")
        plt.show()
        plt.close()

input_data_dir = "data/" # Path to the folder with input images.
out_put_data_dir="generate_data/" # Path to the folder with output images.

#Check if the out_put_data_dir exist, otherwise it will create a new directory with this name
if not os.path.exists(out_put_data_dir):
    os.makedirs(out_put_data_dir)

# Import the Data
img = []
img = [s + ".png" for s in img]

input_images = np.asarray([np.asarray(Image.open(file).resize((im_size, im_size))) for file in glob(input_data_dir + '*') if file not in img])

print ("Input: " + str(input_images.shape))

np.random.shuffle(input_images)

sample_images = random.sample(list(input_images), num_sample)
show_samples(sample_images, out_put_data_dir + "inputs", 0)

# Training loop

D_loss = []
G_loss = []
cum_d_loss = 0
cum_g_loss = 0


print("Training Starts!: ")

# Training loop
for epoch in range(EPOCHS):
    epoch = epoch+ 1

    for batch_images in get_batches(input_images):
        noise_data = np.random.normal(0, 1, size=(batch_size, noise_shape))
        generated_images = generator.predict(noise_data)

        noise_prop = 0.05  # Randomly flip 5% of targets
        real_label = np.zeros(( batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
        flip_idx = np.random.choice(np.arange(len(real_label)), size=int(noise_prop * len(real_label)))
        real_label[flip_idx] = 1 - real_label[flip_idx]

        # Train the discriminator with real data
        d_loss_real = discriminator.train_on_batch(batch_images, real_label)

        # Prepare labels for generated data
        fake_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
        flip_idx = np.random.choice(np.arange(len(fake_labels)), size=int(noise_prop * len(fake_labels)))
        fake_labels[flip_idx] = 1 - fake_labels[flip_idx]

        # Train discriminator on generated data
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        cum_d_loss += d_loss
        D_loss.append(d_loss[0])

        # Train generator
        noise_data = np.random.normal(0, 1, size=(batch_size, noise_shape))
        g_loss = GAN.train_on_batch(noise_data, np.zeros((batch_size, 1)))
        cum_g_loss += g_loss
        G_loss.append(g_loss)

    #plot and print the progress after each epoch
    summarize_epoch(d_loss, g_loss, input_images.shape, epoch, noise_data)