import torch
import sys
from openood.networks import ResNet18_224x224
from openood.networks import ResNet18_32x32 

sys.path.append('..')
sys.path.append('../openood/OpenOOD/')
import numpy as np
#import importlib
#import openood
import torch.nn as nn
from openood.datasets import get_dataloader,get_ood_dataloader
import yaml 
#import openood.datasets
from openood.utils.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ind_name = "imagenet200" #cif10, cif100,imagenet200
num_classes = 200
# load the model
net = ResNet18_224x224(num_classes=num_classes) ##32x32

net.load_state_dict(
    #torch.load( './results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch99_acc0.7810.ckpt', map_location=device)
    #torch.load( './results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s1/best_epoch100_acc0.7710.ckpt', map_location=device)
    #torch.load( './results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s2/best_epoch90_acc0.7760.ckpt', map_location=device)

    #torch.load( './results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch96_acc0.9470.ckpt', map_location=device)
    #torch.load( './results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best_epoch95_acc0.9500.ckpt', map_location=device)
    #torch.load( './results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best_epoch99_acc0.9450.ckpt', map_location=device)


    #torch.load('./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best_epoch89_acc0.8500.ckpt', map_location=device)
    #torch.load('./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s1/best_epoch89_acc0.8560.ckpt', map_location=device)
    torch.load('./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2/best_epoch88_acc0.8480.ckpt', map_location=device)


)
net.cuda()
net.eval()

def get_model_outputs(data_loader, model, layer_num=-2, logits_pos = -1):
    classes = []
    logits = []
    layer_x = []
    for batch in data_loader:
        y = batch["label"]
        x = batch["data"]
        x = x.float()
        #y = y.T[4].long()
        y = y.long()
        classes += y.squeeze(dim=0)
        with torch.no_grad():
            forward_layer_output = model(x.to(device), return_feature=True, return_feature_list=False)
            #print(forward_layer_output[logits_pos].shape)
            logits += forward_layer_output[logits_pos]
            # layer_num=-1 is pre-logits, layer_num=0 is the first layer
            #print(forward_layer_output[layer_num].shape)
            layer_x += forward_layer_output[layer_num]
    output = {
        "logit": torch.stack(logits),
        #"softmax": torch.nn.functional.softmax(torch.stack(logits), dim=1),
        "layer_x": torch.stack(layer_x),
        "classes": torch.stack(classes),
    }

    return output   
#with open("./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config_ood.yml") as stream:
#with open("./results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/config_ood.yml") as stream:
with open('./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/config_ood.yml') as stream:
    try:
        conf = Config(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)


loader_dict = get_dataloader(conf)
#ood_loader_dict = get_ood_dataloader(conf)
#ood_loader_far , ood_loader_near = ood_loader_dict["farood"], ood_loader_dict["nearood"]

ind_dict = list(loader_dict.keys())
#near_ood_dict = list(ood_loader_near.keys())
#far_ood_dict = list(ood_loader_far.keys())

#path = "./cif10_acts_s2"
path = "./imagenet200_acts_s2"
for key in ind_dict:
    print(key)
    buf_data = get_model_outputs(loader_dict[key],net,-1,-2)
    np.save(f"{path}/resnet18_{ind_name}_{key}.npy",buf_data["layer_x"].detach().cpu().numpy())
    np.save(f"{path}/resnet18_{ind_name}_{key}_y.npy",buf_data["classes"].detach().cpu().numpy())
    soft_buf = torch.nn.functional.softmax(buf_data["logit"], dim=1).detach().cpu().numpy()
    np.save(f"{path}/resnet18_{ind_name}_{key}_softs.npy",soft_buf)

#for key in near_ood_dict:
#    print(key)
#    buf_data = get_model_outputs(ood_loader_near[key],net,-1,-2)
#    np.save(f"{path}/resnet18_{ind_name}_{key}.npy",buf_data["layer_x"].detach().cpu().numpy())
#    np.save(f"{path}/resnet18_{ind_name}_{key}_y.npy",buf_data["classes"].detach().cpu().numpy())
#    soft_buf = torch.nn.functional.softmax(buf_data["logit"], dim=1).detach().cpu().numpy()
#    np.save(f"{path}/resnet18_{ind_name}_{key}_softs.npy",soft_buf)


#for key in far_ood_dict:
#    print(key)
#    buf_data = get_model_outputs(ood_loader_far[key],net,-1,-2)
#    np.save(f"{path}/resnet18_{ind_name}_{key}.npy",buf_data["layer_x"].detach().cpu().numpy())
#    np.save(f"{path}/resnet18_{ind_name}_{key}_y.npy",buf_data["classes"].detach().cpu().numpy())
#    soft_buf = torch.nn.functional.softmax(buf_data["logit"], dim=1).detach().cpu().numpy()
#    np.save(f"{path}/resnet18_{ind_name}_{key}_softs.npy",soft_buf)
