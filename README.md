# alexnet
My replication code for the AlexNet paper.


# Data

Im using tiny-imagenet for development. The net was trained on imagenet 2012.

Download tiny-imagenet:
```console
mkdir -p data/tiny-imagenet && wget -O- users.dcc.uchile.cl/~gchapero/datasets/tiny-imagenet.tar | tar -C data -x
```

