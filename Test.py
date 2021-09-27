
import time
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.DPRAEdgeNet import DPRAEdgeNet,EdgeNet
from lib.segnet import SegNet
from lib.pranet import PraNet

from utils.dataloader import test_dataset
import cv2 as cv
parser = argparse.ArgumentParser()
import torchvision.transforms as transforms

from PIL import Image


parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--model', type=str, default="DPRAEdgeNet", help='name of model')


model_path = "/home/debayan/Desktop/MedAI_Project/saved_models/nora_polyp_new_{0}/DPRAEdgeNet-best.pth"

def rgb_loader( path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


for val in range(1,2): 

    ckpt_path = model_path.format(val)
    
    #Test Data Path 
    data_path = '//home/debayan/Desktop/MedAI_Project/new_data/test/MedAI_2021_Polyp_Segmentation_Test_Dataset/'

    #Prediction Save Path
    save_path = '/home/debayan/Desktop/MedAI_Project_nora_final/predictions/nora_polyp_new_{}/'.format(val) 
    opt = parser.parse_args()
    
    if opt.model == "SegNet":
        model = SegNet(input_channels=3, output_channels=1).cuda()
    elif opt.model == "PraNet":
        model = PraNet().cuda()

    elif opt.model == "EdgeNet":
        model =  EdgeNet(128)
    elif opt.model == "DPRAEdgeNet":
        model =  DPRAEdgeNet(128)

    else:

        raise ValueError

    model.load_state_dict(torch.load(ckpt_path),strict=False)
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root   = data_path
    #image_root = '{}/images/'.format(data_path)
    #gt_root = '{}/masks/'.format(data_path)


    test_loader = test_dataset(image_root, None, opt.testsize)
    

    images_org = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
    images_org = sorted(images_org)

    img_transform = transforms.Compose([
                transforms.Resize((opt.testsize, opt.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])


    for i in range(test_loader.size):
        image, gt, gt_edge, name,image_size = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        torch.cuda.synchronize()
        tsince = int(round(time.time()*1000))
        #res,edge = model(image)
        
        #res = model(image)
        if opt.model == "SegNet":
            res,_ = model(image)
        elif opt.model == "PraNet":
            res,_,_,_ = model(image)
        elif opt.model == "EdgeNet" or opt.model == "DPRAEdgeNet":
            res,edge = model(image)
            res = res[0]
        
        
        
        torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time()*1000)) - tsince
        print ('test time elapsed {}ms'.format(ttime_elapsed))
        res = F.upsample(res, size=(image_size[0],image_size[1]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        
        #edge = edge[1].sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        print(save_path+name)
        cv.imwrite(save_path+name, res*255)
        #cv.imwrite(save_edge_path+name, edge*255)
        


