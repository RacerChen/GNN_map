from matplotlib import pyplot

import gmm_main_gpu as gm
import data_preprocessing as dp
from numpy import asarray, load
import numpy as np

# Only use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def use_trained_model_of_dataset(h5_model_name, image_index):
    data = load('maps_256.npz')
    src_images, tar_images = data['arr_0'], data['arr_1']

    g_model = gm.define_generator()
    g_model.load_weights(h5_model_name)
    print(g_model.summary())

    pyplot.subplot(2, 3, 1)
    pyplot.axis('off')
    pyplot.imshow(src_images[image_index].astype('uint8'))

    pyplot.subplot(2, 3, 2)
    pyplot.axis('off')
    pyplot.imshow(tar_images[image_index].astype('uint8'))

    pyplot.subplot(2, 3, 3)
    pyplot.axis('off')

    real_pic = (src_images[image_index] - 127.5) / 127.5
    gen_pic = g_model.predict(np.array([real_pic]))[0]
    gen_pic = (gen_pic * 127.5) + 127.5
    pyplot.imshow(gen_pic.astype('uint8'))

    pyplot.savefig('gen_result.png')
    pyplot.close()


if __name__ == '__main__':
    use_trained_model_of_dataset('/home/jc/GNN_map_proj/pix2pix_model_076720.h5', 1)
