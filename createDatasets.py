from pickle import load, dump
from numpy import ones, zeros, float32, random
with open("clean.pickle","rb") as f:
    cifar, lld = load(f, encoding="bytes")


datasets = zeros(shape=[len(cifar)+len(lld), 32, 32, 3], dtype=float32)
for i, image in enumerate(cifar):
    datasets[i] = image
for j, image in enumerate(lld):
    datasets[i + j] = image

labels = zeros(shape=[len(cifar)+len(lld)], dtype=int)
labels[len(cifar):len(cifar)+len(lld)] = ones(shape=[len(lld)], dtype=int)

del cifar, lld, i, j, image

rng_state = random.get_state()
random.shuffle(datasets)
random.set_state(rng_state)
random.shuffle(labels)


with open("final.pickle", "wb") as o:
    dump((datasets, labels), o)
