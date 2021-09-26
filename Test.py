
import time
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.DRPAEdgeNet import DRPAEdgeNet,EdgeNet
from lib.segnet import SegNet
from lib.pranet import PraNet

from utils.dataloader import test_dataset
import cv2 as cv
parser = argparse.ArgumentParser()
import torchvision.transforms as transforms

from PIL import Image


parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--model', type=str, default="PraNet", help='name of model')


model_path = "/home/debayan/Desktop/MedAI_Project/saved_models/PraNet_cvcclinic_{0}/PraNet-best.pth"

def rgb_loader( path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


for val in range(5): 

    ckpt_path = model_path.format(val)
        
    #Test Data Path 
    data_path = '/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/KVASIR_SEG/'

    #Prediction Save Path
    save_path = '/home/debayan/Desktop/MedAI_Project/predictions/PraNet_cvcclinic_on_kvseg_{}/'.format(val) 
    opt = parser.parse_args()
    
    if opt.model == "SegNet":
        model = SegNet(input_channels=3, output_channels=1).cuda()
    elif opt.model == "PraNet":
        model = PraNet().cuda()

    elif opt.model == "EdgeNet":
        model =  EdgeNet(128)
    elif opt.model == "DRPAEdgeNet":
        model =  DRPAEdgeNet(128)

    else:

        raise ValueError

    model.load_state_dict(torch.load(ckpt_path),strict=True)
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)


    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    

    images_org = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
    images_org = sorted(images_org)

    img_transform = transforms.Compose([
                transforms.Resize((opt.testsize, opt.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])


    for i in range(test_loader.size):
        image, gt, gt_edge, name = test_loader.load_data()
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
        elif opt.model == "EdgeNet" or opt.model == "DRPAEdgeNet":
            res,edge = model(image)
            res = res[0]
        
        
        
        torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time()*1000)) - tsince
        print ('test time elapsed {}ms'.format(ttime_elapsed))
        #res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        #edge = edge[1].sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        print(save_path+name)
        cv.imwrite(save_path+name, res*255)
        #cv.imwrite(save_edge_path+name, edge*255)
        


