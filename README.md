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
   
5. To run GUI:

python app.py
