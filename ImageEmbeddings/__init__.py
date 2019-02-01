from os import environ
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)  # h5py Future Warning
from keras.models import load_model
from keras import backend as K
from numpy import zeros, uint8
from sys import stdout
from pickle import load, loads, dumps
from redis import StrictRedis
from .distanceFunctions import euclideanDistance, cosineDistance
key_name = 'lld_vectors'
icons_path = 'datasets/individuals/'
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Tensorflow compiler Warning


class ImageEmbeddings:
    def __init__(self, mdl_path, verbose=False, load_in_memory=False, layer=13, forceRecal=False):
        self.inform = print if verbose else lambda *a, **k: None

        self.r = StrictRedis()
        self.key_name = key_name
        self.load_in_memory = load_in_memory

        self.ico_path = icons_path
        self.mdl_path = mdl_path
        self.layer = layer
        self.model = load_model(mdl_path)
        self.output_function = K.function([self.model.layers[0].input, K.learning_phase()],
                                          [self.model.layers[layer].output])
        self.layer_info = self.model.layers[layer].get_config()
        self.distances = None

        self.icons = []
        for i in range(5):
            with open('datasets/LLD-icon/LLD-icon_data_' + str(i) + '.pkl', 'rb') as f:
                self.icons.extend(load(f, encoding="bytes"))
                print('File', i + 1, 'of 5 loaded')

        if not self.r.exists(self.key_name) or forceRecal:
            self.calculateVectors()

        self.db_len = self.r.llen(self.key_name)
        self.inform('\nReady to compare images with model:', mdl_path,
                    ' Layer', layer, '- Name:', self.layer_info['name'])

    def calculateVectors(self):
        in_size = len(self.icons)
        print(in_size, 'icons loaded, calculating vectors.')

        self.r.delete(key_name)
        step = 100 / in_size
        for i, icon in enumerate(self.icons):
            vector = self.getOutputFrom(icon)
            self.r.rpush(key_name, dumps(vector))
            stdout.write('\r[IMG-EMB]\tCalculating Vectors {:2.2f}%'.format(step * (i + 1)))

    def getOutputFrom(self, image, mode=0):
        return self.output_function([[image], mode])[0][0]

    def getDBIterator(self):
        if self.load_in_memory:
            return enumerate(self.r.lrange(self.key_name, 0, -1))
        return enumerate(range(self.db_len))

    def calculateDistances(self, img_array, use_euc_dist=True):
        img_vector = self.getOutputFrom(img_array)
        self.inform('Image ready, now calculating distances')

        dists = zeros((self.db_len, 2))
        step = 100 / self.db_len

        for i, vector in self.getDBIterator():
            if not self.load_in_memory:
                vector = self.r.lindex(self.key_name, i)
            if use_euc_dist:
                dists[i] = i, euclideanDistance(img_vector, loads(vector))
            else:
                dists[i] = i, cosineDistance(img_vector, loads(vector))
            stdout.write('\r[IMG-EMB]\tCalculating Distances {:2.2f}%'.format(step * (i + 1)))

        self.inform('\nFinished calculating distances to image')
        self.distances = dists[dists[:, 1].argsort()]

    def viewCollageComparison(self, res_size=10):
        coll = zeros((32, res_size * 32, 3), dtype=uint8)
        self.inform('Distances')
        for i, (idx, dist) in enumerate(self.distances[:res_size]):
            coll[0:32, 0 + i * 32:32 + i * 32] = self.openImage(idx)
            self.inform('Index:', int(idx), 'Distance:', dist)
        return coll

    def openImage(self, idx):
        return self.icons[int(idx)]
