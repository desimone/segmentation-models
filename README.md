# Summary 

⚠️ Work in progress ⚠️

A collection of semantic image segmentation models implemented in TensorFlow. Contains data-loaders for the generic and medical benchmark datasets. 

Hopefully this project will enable researchers to spend less time scaffolding and more time building.


## Datasets & Benchmarks

Generic

- [ ] [ADE20K Scene Parsing](https://groups.csail.mit.edu/vision/datasets/ADE20K/) : [paper](https://arxiv.org/pdf/1608.05442.pdf)
- [ ] [Microsoft COCO: Common Objects in Context](http://mscoco.org/home/) : [paper](https://arxiv.org/abs/1405.0312)
- [ ] [Cityscapes](https://www.cityscapes-dataset.com/) : [paper](https://arxiv.org/abs/1604.01685)
- [ ] [PASCAL Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/) : [paper](https://link.springer.com/article/10.1007/s11263-014-0733-5)
- [ ] [SUN RGB-D Scene Understanding Benchmark Suite](http://rgbd.cs.princeton.edu/) : [paper](http://rgbd.cs.princeton.edu/paper.pdf)

Medical

- [ ] MICCAI - Brain Tumor Image Segmentation Challenge (BRATS)
- [ ] MICCAI - Ischemic Stroke Lesion Segmentation (ISLES)

## Networks & Models

Generic 

- [ ] [DeepLab v2](http://arxiv.org/abs/1412.7062) : [project](http://liangchiehchen.com/projects/DeepLab.html) : [C++ code](https://bitbucket.org/deeplab/deeplab-public/)
- [ ] [RefineNet](https://arxiv.org/abs/1611.06612) : [MATLAB code](https://github.com/guosheng/refinenet)
- [ ] [I-FCN](https://arxiv.org/abs/1611.08986) 
- [ ] [FC-DenseNet](https://arxiv.org/abs/1611.09326) : [theano, lasagne code](https://github.com/SimJeg/FC-DenseNet)
- [ ] [PixelNet](https://arxiv.org/abs/1609.06694) : [cafffe code](https://github.com/endernewton/PixelNet)
- [ ] [FCN](http://arxiv.org/abs/1411.4038) : [slides](https://docs.google.com/presentation/d/1VeWFMpZ8XN7OC3URZP4WdXvOGYckoFWGVN7hApoXVnc) 
- [ ] [SegNet](http://arxiv.org/abs/1505.07293) : [caffe code](https://github.com/alexgkendall/caffe-segnet)

Medical

- [ ] [U-Net](http://lmb.informatik.uni-freiburg.de/resources/opensource/unet.en.html) 

## Usage

See `./scipts/`

## Requirements

- Python 2.7 
- [TensorFlow](https://www.tensorflow.org/get_started/os_setup) `0.12+` 

## Resources

Learn

1. [TensorFlow Deep Learning Course](https://www.udacity.com/course/deep-learning--ud730) Get hands on right away with tensorflow and deep learning.
2. [Machine Learning, Andrew Ng](https://www.coursera.org/learn/machine-learning) Deeper dive into basics, less hands . 
3. [Stanford CS231n](https://cs231n.github.io/) [videos](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) I can't overstate how fantastic the notes, and videos are.  
4. [Deep Learning : Book](http://www.deeplearningbook.org/) Helpful reference for filling in gaps.
5. Above papers, starting with [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) and [video](http://techtalks.tv/talks/fully-convolutional-networks-for-semantic-segmentation/61606/)
 
Code

- [TF-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
- [TF-Slim : Classification Networks](https://github.com/tensorflow/models/tree/master/slim)
- [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)
- [pixel-cnn++](https://github.com/openai/pixel-cnn)
- NVIDIA Digits [Semantic Segmentaiton Example](https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation) [Medical Imaging Example](https://github.com/NVIDIA/DIGITS/tree/master/examples/medical-imaging)

## Contributing

Please do. [PEP-8](https://www.python.org/dev/peps/pep-0008/), [google style](https://google.github.io/styleguide/pyguide.html) with 2 space idents [🤦️](https://www.tensorflow.org/how_tos/style_guide).