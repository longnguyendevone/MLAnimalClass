# About this project
This deep learning project evaluates and compares the performance of various architectures for multi-class image classification. The models classify images into 10 distinct animal species using the Animals-10 dataset. The dataset utilizes approximately 28,000 images split into an 80/20 ratio for training and validation. The implementation heavily focuses on hardware optimization for low-end devices. 
Techniques like Automatic Mixed Precision (AMP) and Gradient Checkpointing are used to ensure efficient training on a 6GB VRAM GPU. 
A Python-based graphical user interface (GUI) built with Gradio is included for real-time user interaction and inference.
# Model Architectures


The repository includes from-scratch implementations of a lightweight Custom Convolutional Neural Network (CNN) and a Custom Vision Transformer (ViT).
The project also features pre-trained models using transfer learning.

The supported pre-trained architectures include ResNet50, EfficientNetV2, MobileNetV3, DeiT, and VGG16.

# How to setup

1. Open Anaconda Prompt:

conda create --name animals10_env python=3.9

conda activate animals10_env

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install matplotlib seaborn scikit-learn pandas tqdm gradio timm

pip install gradio==3.50.2

code --

2. WARNING:

ConvNext and deit_small are not working.

3. Training commands:

    Custom CNN: python main.py --models custom_cnn --epochs 15 --batch_size 32 --lr 0.001
   
    Custom Transformer: python main.py --models custom_transformer --epochs 45 --batch_size 64 --lr 0.0005
   
    MobileNetV3: python main.py --models mobilenet --epochs 30 --batch_size 64 --no_checkpointing
   
    EfficientNetV2: python main.py --models efficientnetv2 --epochs 30 --batch_size 64
   
    ResNet50: python main.py --models resnet50 --epochs 30 --batch_size 64
   
    VGG16: python main.py --models vgg16 --epochs 30 --batch_size 32 --freeze
   
    deit: python main.py --models deit --epochs 30 --batch_size 32 --lr 0.0005
   
4. To run GUI:

python app.py
