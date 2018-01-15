from glob import glob
import pickle
from os.path import join
from numpy import concatenate, array, zeros


def load_icon_data(data_path, pattern='LLD_favicon_data*.pkl',num_files = 6, single_file=None):
    files = glob(join(data_path, pattern))
    files.sort()
    if single_file is None:
        with open(files[0], 'rb') as f:
            icons = pickle.load(f,encoding="bytes")
        if len(files) > 1:
            for file in files[1:num_files]:
                print("Loading data: ",file)
                with open(file, 'rb') as f:
                    icons_loaded = pickle.load(f,encoding="bytes")
                icons = concatenate((icons, icons_loaded))
    else:
        with open(files[single_file % len(files)], 'rb') as f:
            icons = pickle.load(f)
    return icons


# Load LLD
lld = load_icon_data("datasets/LLD", num_files=2)
print("LLD Shape\t\t", lld.shape)

data_path = "datasets/cifar-10/"

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

# Number of files for the training-set.
_num_files_train = 3

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = join(data_path, filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

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


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Convert the images.
    images = _convert_images(raw_images)

    return images


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images from the data-file.
        images_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images


cifar = load_training_data()
print("CIFAR-10 Shape \t", cifar.shape)
with open("clean.pickle","wb") as o:
    pickle.dump((cifar, lld), o)
