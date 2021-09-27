import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.DPRAEdgeNet import DPRAEdgeNet,EdgeNet
from lib.segnet import SegNet 
from lib.pranet import PraNet

from utils.utils import EarlyStopping
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchstat import stat
import glob
import wandb



def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()



def test(model, path,model_name):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/image/'.format(data_path)
    gt_root = '{}/mask/'.format(data_path)

    total_vals = len(glob.glob(gt_root + "/*png"))
    if total_vals == 0 : 
        total_vals = len(glob.glob(gt_root + "/*jpg"))

    test_loader = test_dataset(image_root, gt_root, opt.trainsize)
    b=0.0
    val_loss=0.0

    for i in range(total_vals):
        image, gt, gt_edge,name,_ = test_loader.load_data()

        gt = gt.unsqueeze(0)
        gt_edge = gt_edge.unsqueeze(0)
        image = image.cuda()
        
        if model_name == "PraNet": 
            
            res,t1,t2,t3= model(image)
        
            loss = structure_loss(res, gt.cuda())
            loss1 = structure_loss(t1, gt.cuda())
            loss2 = structure_loss(t2, gt.cuda())
            loss3 = structure_loss(t3, gt.cuda())

            total_loss = loss + loss1 + loss2 + loss3 

        elif opt.model == "DPRAEdgeNet" or opt.model == "EdgeNet":

            seg_map_array, edge_map_array = model(image)
            res = seg_map_array[0]
        
            total_loss  = 0
            for seg_map,edge_map in zip(seg_map_array,edge_map_array): 
                total_loss = total_loss + structure_loss(seg_map, gt.cuda()) + structure_loss(edge_map, gt_edge.cuda())
                

        elif model_name == "SegNet": 

            res,_ = model(image)
            total_loss = structure_loss(res, gt.cuda())

            

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)



        #res = torch.squeeze(res)
        #gt = np.squeeze(gt)


        #res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        val_loss = val_loss + total_loss.item()

        a =  '{:.4f}'.format(loss)
        a = float(a)
        b = b + a
        
    return b/total_vals,val_loss/total_vals

best = 0

def train(train_loader, model, optimizer, scheduler,epoch, test_path,best_meandice,ckpt_folder="harddnet_exp",model_name="Ours"):
    
    model.train()
    # ---- multi-scale training ----
    size_rates = [1]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:

            for param_group in optimizer.param_groups:
            
                lr = param_group['lr']
                wandb.log({'Learning Rate': lr})
            
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts,gts_edge = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            gts_edge = Variable(gts_edge).cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts_edge = F.upsample(gts_edge, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
           

           


            if opt.model == "DPRAEdgeNet" or opt.model == "EdgeNet" :

                seg_map_array, edge_map_array = model(images)
                
                total_loss  = 0
                for seg_map,edge_map in zip(seg_map_array,edge_map_array): 
                    total_loss = total_loss + structure_loss(seg_map, gts) + structure_loss(edge_map, gts_edge)
      
                wandb.log({'Structure loss': total_loss.item()})
      

            
            elif model_name == "SegNet": 
                
                lateral_map_5,_ = model(images)
                total_loss = structure_loss(lateral_map_5, gts)

            elif model_name == "PraNet": 
                
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
                # ---- loss function ----
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)

                total_loss = loss2 + loss3 + loss4 + loss5
            
            
            
            # ---- backward ----
            total_loss.backward()
           
            optimizer.step()
            

            # ---- recording loss ----
            if rate == 1:
                
                loss_record5.update(total_loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                          loss_record5.show()))
    
    save_path = '/home/debayan/Desktop/MedAI_Project/saved_models/{0}/'.format(ckpt_folder)

    os.makedirs(save_path, exist_ok=True)
    
    
    if (epoch+1) % 1 == 0:

        meandice,val_loss = test(model,test_path,model_name)
        wandb.log({'Validation loss': val_loss})
        wandb.log({'Validation mean dice': meandice})


        scheduler.step(val_loss)
        
        with open(save_path + 'dice_per_epoch.txt','a+') as f: 
            f.write("{} {}\n".format(epoch + 1,meandice))
        if meandice > best_meandice:
            best_meandice = meandice
            with open(save_path + 'best_dice.txt','w') as f: 
                f.write("{} {}\n".format(epoch + 1,best_meandice))
            torch.save(model.state_dict(), save_path + '{}-best.pth'.format(model_name))
            print('[Saving Snapshot:]', save_path + '{}-best.pth'.format(model_name),meandice)
    
    return best_meandice,val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=3e-4, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='SGD', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation',
                        default="False", help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')
    
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    parser.add_argument('--train_path', type=str,
                        default='/home/debayan/Desktop/Kvasir-SEG/datasets/HDNetDataset/Kvasir_SEG_Training_880', help='path to train dataset')
    
    parser.add_argument('--val_path', type=str,
                        default='/home/debayan/Desktop/Kvasir-SEG/datasets/HDNetDataset/Kvasir_SEG_Validation_120' , help='path to testing Kvasir dataset')

    parser.add_argument('--ckpt_folder', type=str,
                        default='expt_1', help='type of loss function')
   
    parser.add_argument('--feature_channels', type=int,default=128)

    parser.add_argument('--patience_early_stopping', type=int,default=10)

    parser.add_argument('--patience_scheduler', type=int,default=5)

    parser.add_argument('--model', type=str,default="Ours")

    opt = parser.parse_args()

    # ---- build models ----
    
    if opt.model == "DPRAEdgeNet": 
        
        print("Choosing DPRAEdgeNet")
        model = DPRAEdgeNet(opt.feature_channels).cuda()

    elif opt.model == "EdgeNet": 
        
        print("Choosing EdgeNet")
        model = EdgeNet(opt.feature_channels).cuda()

    elif opt.model == "SegNet": 
        print("Choosing SegNet")
        model = SegNet(input_channels=3, output_channels=1).cuda()
    elif opt.model == "PraNet": 
        print("Using PraNet")
        model = PraNet().cuda()


    es = EarlyStopping(patience=opt.patience_early_stopping)
    params = model.parameters()
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-5, momentum = 0.9)

    wandb.init(project="NORA_final",name=opt.ckpt_folder,reinit=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr = 1e-5,patience=opt.patience_scheduler, verbose=True)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    total_step = len(train_loader)
    
    print("#"*20, "Start Training", "#"*20)

    mean_dice = 0
    for epoch in range(1, opt.epoch):

        mean_dice,val_loss = train(train_loader, model, optimizer, scheduler,epoch, opt.val_path,mean_dice,opt.ckpt_folder,opt.model)
        metric = torch.tensor(val_loss)
        if es.step(metric):
            print("Early Stopping")
            break

