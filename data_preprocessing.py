from os import listdir

from matplotlib import pyplot
from numpy import asarray, load
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import keras

# dataset:　http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
# dataset path
path = '/home/jc/GNN_map_proj/maps_dataset/train/'

# load all images in a directory into memory
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		print(filename)
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]


# 为了让图片在训练的时候加载的快一点，我们把下载的所有的图片都用Numpy保存在maps_256.npz.
def compress_dataset_into():
	# load dataset
	[src_images, tar_images] = load_images(path)
	print('Loaded: ', src_images.shape, tar_images.shape)
	# save as compressed numpy array
	filename = 'maps_256.npz'
	savez_compressed(filename, src_images, tar_images)
	print('Saved dataset: ', filename)


def load_dataset_test():
	# load the dataset
	data = load('maps_256.npz')
	src_images, tar_images = data['arr_0'], data['arr_1']
	print('Loaded: ', src_images.shape, tar_images.shape)
	# plot source images
	n_samples = 3
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(src_images[i].astype('uint8'))
	# plot target image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(tar_images[i].astype('uint8'))
	pyplot.show()


if __name__ == '__main__':
	load_dataset_test()
