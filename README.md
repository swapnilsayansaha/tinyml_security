# Security and Privacy Issues in TinyML Systems
Course Project Repository for ECE 209AS (Secure and Trustworthy Edge Computing Systems), Winter 2022. Members: Swapnil Sayan Saha, Khushbu Pahwa, and Basheer Ammar. Supervisor: Professor Nader SehatBakhsh

## Application and Model Information:
We have three different applications, namely image recognition (dataset: CIFAR 10), audio keyword spotting (dataset: Google Speech Commands), and human activity detection (dataset: AURITUS). We trained several large models and several TinyML models for each applications. The accuracy and number of parameters of each model (flash usage when parameter count unavalable) are listed below.
### Image Recognition

##### Large Models (available through TensorFlow)
- EfficientNetB0 [[paper](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf)]: 93.2%, 4.07M
- EfficientNetB4 [[paper](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf)]: 93.5%, 17.70M
- EfficientNetv2B0 [[paper](http://proceedings.mlr.press/v139/tan21a.html)]: 96.7%, 5.93M
- EfficientNetv2B3 [[paper](http://proceedings.mlr.press/v139/tan21a.html)]: 97.0%, 12.95M
- ResNet50 [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]: 89.7%, 23.62M
- VGG19 [[paper](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/)]: 93.2%, 20.03M

##### TinyML Models
- ResNet8 [[paper](https://openreview.net/pdf?id=8RxxwAut1BI)][[code](https://github.com/mlcommons/tiny)]: 87.1%, 78.67k
- MCUNet DS-CNN (320kB SRAM, 1MB flash) [[paper](https://proceedings.neurips.cc/paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf)][[code](https://github.com/mit-han-lab/tinyml/tree/master/mcunet)]: 87.7%, 0.57M
- MCUNet DS-CNN (256kB SRAN, 1MB flash) [[paper](https://proceedings.neurips.cc/paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf)][[code](https://github.com/mit-han-lab/tinyml/tree/master/mcunet)]: 87.5%, 0.56M

### Audio Keyword Spotting

##### Large Models 
- Attention RNN [[paper](https://arxiv.org/abs/1808.08929)][[code](https://github.com/douglas125/SpeechCmdRecognition)]: 93.1%, 1.29M (used 35 class version instead of 12)
- CNN [[paper](https://arxiv.org/abs/1711.07128)][[code](https://github.com/ARM-software/ML-KWS-for-MCU)]: 82.4%, 94.6k
- GRU [[paper](https://arxiv.org/abs/1711.07128)][[code](https://github.com/ARM-software/ML-KWS-for-MCU)]: 92.2%, 0.50M
- LSTM [[paper](https://arxiv.org/abs/1711.07128)][[code](https://github.com/ARM-software/ML-KWS-for-MCU)]: 92.9%, 1.03M

##### TinyML Models
- DS-CNN [[paper](https://openreview.net/pdf?id=8RxxwAut1BI)][[code](https://github.com/mlcommons/tiny)]: 92.2%, 24.9k
- MicroNets Small DS-CNN [[paper](https://proceedings.mlsys.org/paper/2021/file/a3c65c2974270fd093ee8a9bf8ae7d0b-Paper.pdf)][[code](https://github.com/ARM-software/ML-zoo/tree/master/models/keyword_spotting)]: 84.4%, 114 kB
- MicroNets Large DS-CNN [[paper](https://proceedings.mlsys.org/paper/2021/file/a3c65c2974270fd093ee8a9bf8ae7d0b-Paper.pdf)][[code](https://github.com/ARM-software/ML-zoo/tree/master/models/keyword_spotting)]: 88.8%, 658kB
- TCN [[paper 1](https://link.springer.com/content/pdf/10.1007/978-3-319-49409-8_7.pdf)][[paper 2](https://arxiv.org/abs/1609.03499)][[code](https://github.com/philipperemy/keras-tcn)]: 75.7%, 19.45k


### Human Activity Detection

##### Large Models 
- CNN-LSTM [[paper](https://link.springer.com/content/pdf/10.1007%2F978-981-15-8269-1_4.pdf)][[code](https://github.com/nesl/Robust-Deep-Learning-Pipeline)]: 99.7%, 1.74M
- CNN [[paper](https://link.springer.com/content/pdf/10.1007%2F978-981-15-8269-1_4.pdf)][[code](https://github.com/nesl/Robust-Deep-Learning-Pipeline)]: 99%, 3.03M
- LSTM [[paper](https://link.springer.com/content/pdf/10.1007%2F978-981-15-8269-1_4.pdf)][[code](https://github.com/nesl/Robust-Deep-Learning-Pipeline)]: 97.3%, 201k

##### TinyML Models

- TCN [[paper 1](https://link.springer.com/content/pdf/10.1007/978-3-319-49409-8_7.pdf)][[paper 2](https://arxiv.org/abs/1609.03499)][[code](https://github.com/philipperemy/keras-tcn)]: 93%, 10.55k
- Bonsai [[paper](http://proceedings.mlr.press/v70/kumar17a/kumar17a.pdf)][[code](https://github.com/microsoft/EdgeML)]: 72.6%, 6.31k
- ProtoNN [[paper](http://proceedings.mlr.press/v70/gupta17a/gupta17a.pdf)][[code](https://github.com/microsoft/EdgeML)]:: 72.0%, 6.23k
- FastRNN [[paper](https://proceedings.neurips.cc/paper/2018/file/ab013ca67cf2d50796b0c11d1b8bc95d-Paper.pdf)][[code](https://github.com/microsoft/EdgeML)]:: 98.35, 6.04 kB
- FastGRNN [[paper](https://proceedings.neurips.cc/paper/2018/file/ab013ca67cf2d50796b0c11d1b8bc95d-Paper.pdf)][[code](https://github.com/microsoft/EdgeML)]:: 91.6%, 13.12 kB

