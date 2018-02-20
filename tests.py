from LogoOrCifar import LogoOrCifar
from ImageEmbeddings import ImageEmbeddings
las_buenas = [8, 10, 150, 1150, 21]

model = LogoOrCifar(cifar_len=0, lld_len=0, train_sz=100000)
if model.defineModel(conv=2, flat=1024):
    model.fitToData(batch_size=128, epochs=1)
mdl_fna = model.model_name
del model
img_emb = ImageEmbeddings(mdl_fna=mdl_fna, layer=13)
img_emb.compareFor(targets=las_buenas)
