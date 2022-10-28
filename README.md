# alexnet
My replication code for the [AlexNet paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html). 


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

Instructions on how to configure `kaggle` are [here](https://github.com/Kaggle/kaggle-api)


## Summary of features / techniques used in the paper

- [x] for rect images, first rescale so that shorter side is of 256 pix, then
  crop longer side to 256 pix, always square images. 
- [x] substract "mean activity" from each pixel, apparently this is mean
  activity per channel
- [x] ReLU non linearity everywhere
- [x] training on multiple gpus (see section 3.2 and fig 2). I wil not do this,
  instead one can leverage the `groups` option in `nn.Conv2d` so emulate the
  behaviour
- [x] local response norm, pytorch implementation divides alpha by n, so in
  order to replicate paper alpha should be multiplied by n
- [x] precise net description taken from
  https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-2gpu-model.cfg
- [x] augmentation: at train time, extract random 224 x 224 patches and
  horizontal reflection
- [x] augmentation: at test time, extract 5 224 x 224 patches (corners +
  center) + h reflections, results in averaged prediction over the 10 patches
- [x] augmentation: PCA color augmentation, see paper section 4.1, extra resources:
	* https://github.com/koshian2/PCAColorAugmentation
	* https://github.com/aparico/pca-color-augment
	* https://groups.google.com/g/lasagne-users/c/meCDNeA9Ud4?pli=1
- [x] dropout

## Training details

- [ ] optim SGD, see exact update rule in paper section 5 (possibly should implement myself)
- [ ] batch size 128
- [ ] dropout of 0.5
- [ ] momentum 0.9
- [ ] weight decay 0.0005
- [ ] weight init is guassian with mean 0, std 0.01. Bias in 2, 4, 5 conv layers and also fc layers with constant 1, bias in other layers init with 0
- [ ] same lr for al layers, start at 0.01
- [ ] decay learn rate by factor of 10 when val error stops improving. This results in 3 reductions during training.
- [ ] 90 epochs, which took 5/6 days.

