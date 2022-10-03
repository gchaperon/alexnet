# alexnet
My replication code for the AlexNet paper.


## Data
The data used is ImageNet 2012. I downloaded it from kaggle.  The paper also
experiments with ImageNet 2010 but I couldn't find this dataset.  The authors
also experiment with pretrainind on ImageNet Fall 2011 which isn't even
available anymore. The closest would be current ImageNet21K, but I don't have
enough compute for that.

```console
mkdir -p data/imagenet && cd data/imagenet
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip
```
Im using tiny-imagenet for development. The net was trained on imagenet 2012.

Download tiny-imagenet:
```console
mkdir -p data/tiny-imagenet && wget -O- users.dcc.uchile.cl/~gchapero/datasets/tiny-imagenet.tar | tar -C data -x
```

## Summary of features / techniques used in the paper

* for rect images, first rescale so that shorter side is of 256 pix, then crop longer side to 256 pix, always square images
* substract "mean activity" from each pixel, apparently this is mean activity per channel
* ReLU non linearity everywhere
* training on multiple gpus (see section 3.2 and fig 2). I wil not do this, instead one can leverage the `groups` option in `nn.Conv2d` so emulate the behaviour
* local response norm, should be implemented from scratch, since paper description doesn't match `nn.LocalResponseNorm` exactly
* precise net description taken from https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-2gpu-model.cfg
* [augmentation] at train time, extract random 224 x 224 patches and horizontal reflection
* [augmentation] at test time, extract 5 224 x 224 patches (corners + center) + h reflections, results in averaged prediction over the 10 patches
* [augmentation] PCA color augmentation, see paper section 4.1, extra resources:
	* https://github.com/koshian2/PCAColorAugmentation
	* https://github.com/aparico/pca-color-augment
	* https://groups.google.com/g/lasagne-users/c/meCDNeA9Ud4?pli=1
* dropout

## Training details

* optim SGD, see exact update rule in paper section 5 (possibly should implement myself)
* batch size 128
* momentum 0.9
* weight decay 0.0005
* weight init is guassian with mean 0, std 0.01. Bias in 2, 4, 5 conv layers and also fc layers with constant 1, bias in other layers init with 0
* same lr for al layers
* decay learn rate by factor of 10 when val error stops improving. This results in 3 reductions during training.
* 90 epochs, which took 5/6 days.

