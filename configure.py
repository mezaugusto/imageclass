import argparse
from pickle import dump
if __name__ == '__main__':
    p = argparse.ArgumentParser("Configure ImageClass")
    p.add_argument("--model", default="saved_models/cifar_0_lld_50000_flat_1024_conv_2.h5",
                   action="store", dest="model",
                   help="Model Path [saved_models/cifar_0_lld_50000_flat_1024_conv_2.h5]")
    p.add_argument("--layer", default=13, type=int,
                   action="store", dest="layer",
                   help="Objective layer [13]")
    p.add_argument("--memory", default=False,
                   action="store", dest="memory",
                   help="Load dataset in memory (Faster) [False]")
    p.add_argument("--verbose", default=False,
                   action="store", dest="verbose",
                   help="Verbose output [False]")
    p.add_argument("--force", default=False,
                   action="store", dest="forceRecal",
                   help="Force vector recalculation [False]")

    opts = p.parse_args()
    dump(opts, open("opts.conf", 'wb'))
