from ImageEmbeddings import ImageEmbeddings
from celery import Task, Celery
from cairosvg import svg2png
from numpy import array
from PIL import Image
from pickle import load
from base64 import b64encode
from io import BytesIO
from tempfile import NamedTemporaryFile


def svg2array(bytestring, size=32, tran_color=255):
    f = NamedTemporaryFile()
    svg2png(bytestring=bytestring, write_to=f)
    img = Image.open(f)
    img = array(img.resize((size, size)))
    img[img[:, :, 3] == 0] = [tran_color, tran_color, tran_color, 0]
    f.close()
    return img[:, :, 0:3]


def array2bytes(arr):
    img = Image.fromarray(arr)
    byte_io = BytesIO()
    img.save(byte_io, 'JPEG')
    im_data = byte_io.getvalue()
    return 'data:image/jpeg;base64,' + b64encode(im_data).decode()


class getSuggestions(Task):
    _img_emb = None

    def __init__(self):
        self.opts = load(open('opts.conf', 'rb'))

    @property
    def img_emb(self):
        if self._img_emb is None:
            self._img_emb = ImageEmbeddings(self.opts.model, verbose=self.opts.verbose, load_in_memory=self.opts.memory,
                                            layer=self.opts.layer, forceRecal=self.opts.forceRecal)
        return self._img_emb

    def run(self, data):
        self.img_emb.calculateDistances(svg2array(data))
        return array2bytes(self.img_emb.viewCollageComparison(res_size=20))


app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
getSug = getSuggestions()
app.register_task(getSug)
