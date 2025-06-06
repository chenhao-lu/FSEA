# FSEA
We used the TransferAttack framework https://github.com/Trustworthy-AI-Group/TransferAttack Same dataset and interface implementation

Dataset Introduction: Randomly sample 1,000 images from ImageNet validate set, in which each image is from one category and can be correctly classified by the adopted models (For some categories,  cannot choose one image that is correctly classified by all the models. In this case, select the image that receives accurate classifications from the majority of models.

Dataset:
https://huggingface.co/datasets/Trustworthy-AI-Group/TransferAttack/blob/main/data.zip
Extract datazip to/path/to/path for use

The commands for training and evaluation are as follows

python fsea_attack.py --input_dir ./path/to/data --output_dir adv_data/attack/ensemble --model='resnet18,inception_v3,deit_tiny_patch16_224,swin_tiny_patch4_window7_224'

fsea_attack.py --input_dir ./path/to/data --output_dir adv_data/attack/ensemble --eval
