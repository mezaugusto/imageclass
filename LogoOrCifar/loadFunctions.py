from glob import glob
from os import makedirs, getcwd
from os.path import join, isdir
from numpy import array, zeros, ones, float32, concatenate
from numpy.random import seed, shuffle, get_state, set_state
from pickle import load, dump
from sys import stdout


seed(42)
# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes for CIFAR
num_classes = 10

# Number of images for each batch-file in CIFAR
_images_per_file = 10000


def _cut_array(arr, lenght):
    if 0 < lenght < len(arr):
        shuffle(arr)
        return arr[:lenght]
    return arr


def _unpickle(filename, data_path):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = join(data_path, filename)

    print('Loading:' + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename, data_path):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename, data_path)

    # Get the raw images.
    raw_images = data[b'data']

    # Convert the images.
    images = _convert_images(raw_images)

    return images


def load_cifar_data(cifar_len, data_path='datasets/cifar-10/'):
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """
    _num_files_train = 6

    # Total number of images in the training-set.
    # This is used to pre-allocate arrays for efficiency.
    _num_images_train = _num_files_train * _images_per_file

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images from the data-file.
        images_batch = _load_data('data_batch_' + str(i + 1), data_path)

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return _cut_array(images, cifar_len)


def load_lld_data(lld_len, data_path='datasets/LLD', single_file=None):
    pattern = 'LLD_favicon_data*.pkl'
    num_files = 6
    files = glob(join(data_path, pattern))
    files.sort()
    if single_file is None:
        with open(files[0], 'rb') as f:
            icons = load(f, encoding='bytes')
        if len(files) > 1:
            for file in files[1:num_files]:
                print("Loading:", file)
                with open(file, 'rb') as f:
                    icons_loaded = load(f, encoding='bytes')
                icons = concatenate((icons, icons_loaded))
    else:
        with open(files[single_file % len(files)], 'rb') as f:
            icons = load(f)
    return _cut_array(icons, lld_len)


def createDataset(cifar, lld):
    print('Creating Dataset')
    dataset_size = len(cifar) + len(lld)
    step = 100 / dataset_size
    images = zeros(shape=[dataset_size, 32, 32, 3], dtype=float32)
    for i, image in enumerate(cifar):
        images[i] = image
        stdout.write('\rProcessing {:2.2f}%'.format(step * (i + 1)))
    for j, image in enumerate(lld):
        images[i + j] = image
        stdout.write('\rProcessing {:2.2f}%'.format(step * (i + j + 1)))

    labels = zeros(shape=[dataset_size], dtype=int)
    labels[len(cifar):dataset_size] = ones(shape=[len(lld)], dtype=int)

    del cifar, lld
    print('\nDataset Created, Shuffling (this can take a while)...')
    rng_state = get_state()
    shuffle(images)
    set_state(rng_state)
    shuffle(labels)
    return images, labels


def loadAndCreateDataset(cifar_len, lld_len):
    print('Loading LLD Data')
    lld = load_lld_data(lld_len)
    print('Loading CIFAR Data')
    cifar = load_cifar_data(cifar_len)
    print('Datasets Loaded, LLD Shape:', lld.shape, 'CIFAR Shape:', cifar.shape)
    return createDataset(cifar, lld)


def createDirs(rel_path):
    save_dir = join(getcwd(), rel_path)
    if not isdir(save_dir):
        makedirs(save_dir)
    return save_dir


def expandPickle():
    for i in range(5):
        with open('datasets/LLD/LLD_favicon_data_' + str(i) + '.pkl', 'rb') as f:
            if i == 0:
                icons = load(f, encoding="bytes")
            else:
                icons = concatenate((icons, load(f, encoding="bytes")))
            print('File', i + 1, 'of 5 loaded')

    step = 100 / len(icons)
    for i, img in enumerate(icons):
        with open('datasets/individuals/img_' + str(i), 'wb') as file:
            dump(img, file)
        stdout.write('\rProcessing {:2.2f}%'.format(step * (i + 1)))