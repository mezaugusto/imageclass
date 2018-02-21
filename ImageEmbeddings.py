from keras.models import load_model
from keras import backend as K
from numpy import zeros, concatenate, uint8, array, asarray, searchsorted, insert
from scipy.spatial.distance import euclidean
from sys import stdout
from PIL.Image import fromarray as arr2img, open as open_img
from _pickle import load, dumps, loads
import redis

r = redis.StrictRedis(host='localhost', port=6379, db=0)
key_name = 'lld_vectors'


class ImageEmbeddings:
    def __init__(self, res_pth='results/', mdl_fna='keras_trained_model3.h5', mode=0, layer=13, forceRecal=False,
                 list_size=20):
        self.res_pth = res_pth
        self.mdl_fna = mdl_fna
        self.mode = mode
        self.layer = layer
        self.list_size = list_size
        self.distances = []
        self.model = load_model('saved_models/' + mdl_fna)
        self.getOutputFrom = K.function([self.model.layers[0].input, K.learning_phase()],
                                [self.model.layers[layer].output])
        self.layer_info = self.model.layers[layer].get_config()
        if not r.exists(key_name) or forceRecal:
            self.calculateVectors()
        self.icons = []
        for i in range(5):
            with open('datasets/LLD/LLD_favicon_data_' + str(i) + '.pkl', 'rb') as f:
                if i == 0:
                    self.icons = load(f, encoding="bytes")
                else:
                    self.icons = concatenate((self.icons, load(f, encoding="bytes")))
                print('File', i+1, 'of 5 loaded')
        print('\nReady to compare images with model:', mdl_fna,
              ' Layer', layer, '- Name:', self.layer_info['name'])

    def calculateVectors(self):
        in_size = len(self.icons)
        print(in_size, 'icons loaded, calculating vectors.')

        step = 100 / in_size
        r.delete(key_name)
        for i, icon in enumerate(self.icons):
            r.lpush(key_name, dumps(self.getOutputFrom([[icon], self.mode])[0][0]))
            stdout.write('\rProcessing {:2.2f}%'.format(step * (i + 1)))

    def calculateDistances(self, img_fname='test.jpg'):
        img = open_img(img_fname)
        vectors = r.lrange(key_name, 0, -1)
        img_vector = self.getOutputFrom([[array(img)], self.mode])[0][0]
        print('Image ready, now calculating distances')

        dists = zeros((self.list_size, 2))
        step = 100 / len(vectors)
        for i, vector in enumerate(vectors):
            vector = loads(vector)
            if i < self.list_size:
                dists[i] = i, euclidean(img_vector, vector)
                if i == self.list_size - 1:
                    dists = dists[dists[:, 1].argsort()]
            else:
                distance = euclidean(img_vector, vector)
                if distance < dists[-1][1]:
                    idx = searchsorted(dists[:, 1], distance)
                    dists = insert(dists, idx*2, (i, distance))[:-2]
                    dists = dists.reshape((self.list_size, 2))
            stdout.write('\rProcessing {:2.2f}%'.format(step * (i + 1)))
        print('\nFinished calculating distances to image')
        self.distances = dists

    def viewCollageComparison(self):
        coll = zeros((len(self.distances) * 32, 32, 3), dtype=uint8)
        for i, (idx, dist) in enumerate(self.distances):
            coll[0 + i * 32:32 + i * 32] = self.icons[int(idx)]
        print('Distances')
        print(self.distances)
        self.showAndSaveImage(coll)

    def showAndSaveImage(self, matrix):
        img = arr2img(matrix)
        img.show()
        img.save(self.res_pth + 'model_' + self.mdl_fna + '.jpg')
