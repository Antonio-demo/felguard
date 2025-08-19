# felguard
# Requirement

Python=3.9

pytorch=1.10.1

scikit-learn=1.0.2

opencv-python=4.5.5.64

Scikit-Image=0.19.2

matplotlib=3.4.3

hdbscan=0.8.28

jupyterlab=3.3.2

Install instructions are recorded in install_requirements.sh

# Run

VGG and ResNet18 can only be trained on CIFAR-10 dataset, while CNN can only be trained on the fashion-MNIST dataset.

Quick start:

```
python main_fed.py --defence felguard --model resnet --dataset cifar --local_ep 1 --local_bs 500 --attack badnet --triggerX 27 --triggerY 27 --epochs 500 --poison_frac 0.5
```

It costs more than 10 GPU hours to run this program.

Detailed settings:

```
python main_fed.py      --dataset cifar,fashion_mnist \
                        --model VGG,resnet,cnn \
                        --attack badnet, dba, adaptive, labelflip, biasattack, layerattack, flipupdate\
                        --lr 0.1 \
                        --malicious 0.1 \
                        --poison_frac 0.5 \
                        --local_ep 1 \
                        --local_bs 500, 600 \
                        --attack_begin 0 \
                        --defence avg, fldetector, fltrust, flame, krum, RLR, felguard \
                        --epochs 500 \
                        --attack_label 5 \
                        --attack_goal -1 \
                        --trigger 'square','pattern','watermark','apple' \
                        --triggerX 27 \
                        --triggerY 27 \
                        --gpu 0 \
                        --save save/your_experiments \
                        --iid 0,1 
```

Images with triggers on the attack process and test process are shown in './save' when running. Results files are saved in './save' by default, including a figure and an accuracy record. More default parameters on different defense strategies or attack can be seen in './utils/options'.
