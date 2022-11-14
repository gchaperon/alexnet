# alexnet
My replication code for the [AlexNet paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html). 

tldr: go to [Results](#results)

## Data
The data used is ImageNet 2012. I downloaded it from kaggle.  The paper also
experiments with ImageNet 2010 but I couldn't find this dataset.  The authors
also experiment with pretraining on ImageNet Fall 2011 which isn't even
available anymore. The closest would be current ImageNet21K, but I don't have
enough compute for that.

Download and extract ImangeNet
```console
mkdir -p data/imagenet && cd data/imagenet
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip
```

Instructions on how to configure `kaggle` are [here](https://github.com/Kaggle/kaggle-api)

Also, if you want to train on tiny-imagenet download the data as follows.
```console
mkdir -p data && cd data
wget https://image-net.org/data/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
cd ..
```

I coded a couple more tasks, but those are downloaded automatically.

## User guide
Install the deps found in `requirements.txt`. I used python 3.9 and pytorch
1.12. You should modify the cuda version according to your hardware.

These are the command line options
```console
$ python -m alexnet --help
Usage: python -m alexnet [OPTIONS]

Options:
  --task [mnist|fashion-mnist|cifar10|cifar100|tiny-imagenet|imagenet]
                                  [required]
  --batch-size INTEGER            [default: 128]
  --dropout FLOAT                 [default: 0.5]
  --learn-rate FLOAT              [default: 0.0001]
  --seed INTEGER                  [default: 12331]
  --extra-logging                 Whether to log histograms of parameters and
                                  grads.
  --fast-dev-run                  Run only a couple of steps, to check if
                                  everything is working properly.
  --help                          Show this message and exit.
```

The available tasks can be seen above. Default hparams where chosen according
to the paper and my own experimentation. To run a setup pretty close to the one
on the paper simply run

```console
python -m alexnet --task imagenet
```

## Results

See the training curves in the [tensorboard.dev](https://tensorboard.dev/experiment/dKB37FAMRwukVz1Sv4QCQA/) experiment.

All experiments were done using an RTX3090. ImageNet training took ~3 days to reach 100 epochs
(hardware has come a long way since 2012, compare with the 5 days it took on an RTX580).

Below are the results on different tasks.

|         | imagenet<br/>(my experiment) | imagenet<br/>(paper) | mnist | fashion-mnist | cifar10 | cifar100 | tiny-imagenet |
|---------|--------------------------|------------------|-------|---------------|---------|----------|---------------|
| error@1 | 0.48                     | 0.40             | 0.01  | 0.09          | 0.13    | 0.41     | 0.57          |
| error@5 | 0.25                     | 0.18             | 0.00  | 0.00          | 0.01    | 0.16     | 0.32          |


Results aren't that close, but also not that far off. As we can see in the training curves, we might have been able to obtain a better results with more training time, since val error seemed to be going down still when training stoped. For this scope however it's good enough.


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
- [x] net description taken from section 3.5 and figure 2
- [x] augmentation: at train time, extract random 224 x 224 patches and
  horizontal reflection
- [x] augmentation: at test time, extract 10 224 x 224 patches (corners +
  center) + h reflections, results is averaged prediction over the 10 patches
- [x] augmentation: PCA color augmentation, see paper section 4.1, extra resources:
	* https://github.com/koshian2/PCAColorAugmentation
	* https://github.com/aparico/pca-color-augment
	* https://groups.google.com/g/lasagne-users/c/meCDNeA9Ud4?pli=1
- [x] dropout

## Training details

- [ ] optim SGD, see exact update rule in paper section 5 (not used see, notes below)
- [x] batch size 128
- [x] dropout of 0.5
- [ ] momentum 0.9 (see notes about optimization below)
- [ ] weight decay 0.0005 (see notes about optimzation below)
- [x] weight init is guassian with mean 0, std 0.01. Bias in 2, 4, 5 conv layers and also fc layers with constant 1, bias in other layers init with 0
- [ ] same lr for al layers, start at 0.01 (close, but see notes)
- [x] decay learn rate by factor of 10 when val error stops improving. This results in 3 reductions during training.
- [x] 90 epochs, which took 5/6 days.

## Notes

* Optimizer changed, [other
  user](https://github.com/dansuh17/alexnet-pytorch/blob/8aeaedd79a462b79d5d483d6774e344ca1738570/model.py#L142)
  also saw poor performance using `optim.SGD`. I even implemented the actual
  optimizer step described in the paper, since it's a little bit different than
  pytorch's algortihm, but I saw no improvement. I kept Adam optim and used a
  learn rate of 1e-4, and also lowered it on plateau.
* Apparently the net convergence is super sensitive to param initialization.
  Out of three seed values I tried only one made the net learn in the imagenet experiment (using the
  exact same hparams elsewhere). The current default seed is the one I found.
