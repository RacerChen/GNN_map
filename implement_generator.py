from matplotlib import pyplot
from PIL import ImageGrab, Image
import gmm_main_gpu as gm
import data_preprocessing as dp
from numpy import asarray, load
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# Only use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def use_trained_model_of_dataset(h5_model_name, image_index):
    data = load('maps_256.npz')
    src_images, tar_images = data['arr_0'], data['arr_1']

    g_model = gm.define_generator()
    g_model.load_weights(h5_model_name)
    print(g_model.summary())

    pyplot.subplot(1, 3, 1)
    pyplot.axis('off')
    pyplot.imshow(src_images[image_index].astype('uint8'))

    pyplot.subplot(1, 3, 2)
    pyplot.axis('off')
    pyplot.imshow(tar_images[image_index].astype('uint8'))

    pyplot.subplot(1, 3, 3)
    pyplot.axis('off')

    real_pic = (src_images[image_index] - 127.5) / 127.5
    gen_pic = g_model.predict(np.array([real_pic]))[0]
    gen_pic = (gen_pic * 127.5) + 127.5
    pyplot.imshow(gen_pic.astype('uint8'))

    pyplot.savefig('gen_result_dataset.png')
    pyplot.close()


def use_trained_model_of_given_img(h5_model_name, image_name):
    # img = Image.open(image_name)
    # img = img.resize((img.size[0], img.size[1]), Image.ANTIALIAS)

    pixels = load_img(image_name, target_size=(256, 256))
    # convert to numpy array
    pixels = img_to_array(pixels)

    g_model = gm.define_generator()
    g_model.load_weights(h5_model_name)
    print(g_model.summary())

    pyplot.subplot(1, 2, 1)
    pyplot.axis('off')
    pyplot.imshow(pixels.astype('uint8'))

    pyplot.subplot(1, 2, 2)
    pyplot.axis('off')
    real_pic = (pixels - 127.5) / 127.5
    gen_pic = g_model.predict(np.array([real_pic]))[0]
    gen_pic = (gen_pic * 127.5) + 127.5
    pyplot.imshow(gen_pic.astype('uint8'))

    pyplot.savefig('gen_result_givenimg.png')
    pyplot.close()


if __name__ == '__main__':
    model_path = '/home/jc/GNN_map_proj/pix2pix_model_109600.h5'
    # use_trained_model_of_dataset(model_path, 1)
    use_trained_model_of_given_img(model_path, '/home/jc/GNN_map_proj/tongji_jd.png')
