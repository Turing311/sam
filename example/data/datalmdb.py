
import torch
import torch.utils.data as data
import numpy as np
import lmdb
import random
import data.caffe_pb2 as caffe_pb2

from skimage.io import imread
from skimage.transform import resize

#DataLmdb("aaa-train", db_size=, crop_size=128, flip=True, scale=0.00390625)
#DataLmdb("aaa-valid", db_size=, crop_size=128, flip=False, scale=0.00390625, random=False)

class DataLmdb(data.Dataset):

    def __init__(self, path, db_size, crop_size, flip, scale, random = True):
        self.path = path
        self.db_size = db_size
        self.crop_size = crop_size
        self.flip = flip
        self.scale = scale
        self.random = random
        self.lmdb_env = lmdb.open(self.path, readonly=True)
        self.lmdb_txn = self.lmdb_env.begin()
        self.lmdb_cursor = self.lmdb_txn.cursor()

    def __len__(self):
        return self.db_size

    def __getitem__(self, index):
        """Returns the i-th example."""
        key = self.lmdb_cursor.key()
        if len(key) == 0:
            self.lmdb_cursor.first()

        datum = caffe_pb2.Datum()
        datum.ParseFromString(self.lmdb_cursor.value())
        self.lmdb_cursor.next()

        w = datum.width
        h = datum.height
        c = datum.channels
        y = datum.label

        xint8 = np.fromstring(datum.data, dtype=np.uint8).reshape(c, h, w)

        if self.random:
            top = random.randint(0, h - self.crop_size - 1)
            left = random.randint(0, w - self.crop_size - 1)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2

        bottom = top + self.crop_size
        right = left + self.crop_size
        xint8 = xint8[:, top:bottom, left:right]

        if self.flip:
            if random.randint(0, 1):
                xint8 = xint8[:, :, ::-1]

        xf32 = xint8 * np.float32(self.scale)
        return xf32, y
