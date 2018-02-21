from ImageEmbeddings import ImageEmbeddings

mdl_fna = "cifar_0_lld_50000_flat_1024_conv_2.h5"
img_emb = ImageEmbeddings(mdl_fna=mdl_fna, layer=13, forceRecal=False)
img_emb.calculateDistances()
img_emb.viewCollageComparison()
