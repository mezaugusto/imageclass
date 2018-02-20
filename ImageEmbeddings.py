from keras.models import load_model
from keras import backend as K
from numpy import zeros, concatenate, uint8
from scipy.spatial.distance import euclidean
from sys import stdout
from PIL.Image import fromarray as arr2img
from os.path import exists
from pickle import dump, load
from os import remove


class ImageEmbeddings:
    def __init__(self, res_pth='results/', mdl_fna='keras_trained_model3.h5', mode=0, layer=13, forceRecal=False):
        self.res_pth = res_pth
        self.mdl_fna = mdl_fna
        self.mode = mode
        self.layer = layer
        self.icons = None
        self.outputs = None
        self.forceRecal = forceRecal
        self.model = load_model('saved_models/' + mdl_fna)
        self.getOutputFrom = K.function([self.model.layers[0].input, K.learning_phase()],
                                [self.model.layers[layer].output])
        self.layer_info = self.model.layers[layer].get_config()
        print('Ready to compare LLD images with model:', mdl_fna,
              ' Layer', layer, '- Name:', self.layer_info['name'])

    def compareFor(self, targets):
        for i in range(5):
            with open('datasets/LLD/LLD_favicon_data_' + str(i) + '.pkl', 'rb') as f:
                if i == 0:
                    self.icons = load(f, encoding="bytes")
                else:
                    self.icons = concatenate((self.icons, load(f, encoding="bytes")))

        print('Processing Files:')
        self.calculateVectors()
        for target in targets:
            distances = self.calculateDistances(target)
            self.viewCollageComparison(distances[:20], target)

    def calculateVectors(self):
        in_size = len(self.icons)
        out_size = self.model.layers[self.layer].output.shape[1]
        self.outputs = zeros((in_size, out_size))
        print(in_size, 'icons loaded, calculating vectors.')

        step = 100 / in_size
        for i, icon in enumerate(self.icons):
            self.outputs[i] = self.getOutputFrom([[icon], self.mode])[0][0]
            stdout.write('\rProcessing {:2.2f}%'.format(step * (i + 1)))

    def calculateDistances(self, target):
        print('Ready, now calculating distances')
        print('Targeting icon', target)

        distances = zeros((self.outputs.shape[0], 2))
        distances[target][1] = 1000000
        step = 100 / self.outputs.shape[0]

        for i, out in enumerate(self.outputs):
            if target != i:
                distances[i] = i, euclidean(self.outputs[target], out)
            stdout.write('\rProcessing {:2.2f}%'.format(step * (i + 1)))

        print('\nFinished calculating distances to image', target)
        return distances[distances[:, 1].argsort()]

    def viewCollageComparison(self, rows, target):
        coll = zeros((len(rows) * 32, 64, 3), dtype=uint8)
        img1 = self.icons[target]
        for i, (idx, dist) in enumerate(rows):
            comparison = concatenate((img1, self.icons[int(idx)]), axis=1)
            coll[0 + i * 32:32 + i * 32] = comparison
        print('Minimum distance between them:', rows[0][1])
        self.showAndSaveImage(coll,target)

    def showAndSaveImage(self, matrix,target):
        img = arr2img(matrix)
        img.show()
        img.save(self.res_pth + 'target_'+ str(target) + 'model_' + self.mdl_fna + '.jpg')
