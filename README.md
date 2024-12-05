## A GAN-generated image detection algorithm based on spectrum and border features

This is a PyTorch implementation of BorderF

### Requirements
* PyTorch >= 1.9.0
* Python >= 3.7.0

## (3) Dataset

The wang dataset can be found in `https://peterwang512.github.io/CNNDetection/`

The faceshq dataset can be found in `https://drive.google.com/file/d/1AqbGw82ueBP3fNNVCbXZgOPPFsh2uNXm/view`
After downloading the faceshq dataset, please modify the corresponding datasets to Stylegan and Stylegan2, and name the corresponding real dataset and generated dataset as 0_real and 1_fake, respectively.

### How to run
Please change the dataset path in  `config.py`  to your dataset path

then
for train:
python train.py

for test:
python test.py ,  if test the faceshq, please  modify the code of line 3 and line 4 on `test.py`.

## (A) Acknowledgments
Our code implementation draws inspiration from `https://github.com/EricGzq/GocNet-pytorch`