import glob 
from metrics import  Dice_Coefficient, All_Hausdorff_Distances_Binary_Image
import os
import cv2 as cv
import numpy as np
import torch
from sklearn.metrics import precision_score,f1_score,recall_score,jaccard_score
from tqdm import tqdm

path = "/home/debayan/Desktop/MedAI_Project/predictions"
gt_path = "/home/debayan/Desktop/Kvasir/datasets/final_datasets/{}/{}"
metrics_txt = "/home/debayan/Desktop/MedAI_Project/results/"

try: 

    os.makedirs(metrics_txt)

except: 

    print("Dir exists")

metrics_txt = metrics_txt + "results_cvcclinic.txt"

datasets = sorted(os.listdir(path))

num_classes = 2
threshold = 0.5
size = 256
dice_coeff = Dice_Coefficient()
PARAMS = {'HD_IGNORE_CLASS_INDEX': 0, 'IGNORE_FAILED_HD':False}


hauss_d = All_Hausdorff_Distances_Binary_Image(PARAMS)
test_datset_loc = "/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/{}/test/{}/*png" 
path_model =  path 

model_preds = sorted(os.listdir(path_model))
#print(model_preds)

for mod in tqdm(model_preds,total=len(model_preds)): 

    print("Calculating for ",mod)
    val_images = test_datset_loc.format("CVC-ClinicDB","images")
    val_masks = test_datset_loc.format("CVC-ClinicDB","masks") 
    
    """
    if "sess" in mod:
        print("HERE")
        
        val_images = test_datset_loc.format("KVASIR_SESS","images")
        val_images = val_images.replace("png","jpg")
        val_masks = test_datset_loc.format("KVASIR_SESS","masks") 
        val_masks = val_masks.replace("png","jpg")
    
    elif "kvseg_" in mod:
        
        print("HERE 1")
        val_images = test_datset_loc.format("KVASIR_SEG","images")
        val_masks = test_datset_loc.format("KVASIR_SEG","masks") 
    
    
    elif "cvcclinic" in mod: 
        print("HERE")
        val_images = test_datset_loc.format("CVC-ClinicDB","images")
        val_masks = test_datset_loc.format("CVC-ClinicDB","masks") 

    elif "cvccolon" in mod: 
        print("HERE")
        val_images = test_datset_loc.format("CVC-ColonDB","images")
        val_masks = test_datset_loc.format("CVC-ColonDB","masks")

    elif "etis" in mod: 
        print("HERE")
        val_images = test_datset_loc.format("ETIS-LaribPolypDB","images")
        val_masks = test_datset_loc.format("ETIS-LaribPolypDB","masks")

    """
    
    
    
    gt_imgs = sorted(glob.glob(val_masks))
    pred_path = path + "/" + mod + "/*png"
    pred_imgs = sorted(glob.glob(pred_path))

    model_dice = []  
    model_iou = []  
    model_dhd = []  
    model_mhd = [] 
    model_prec = []  
    model_recall = []  
    model_f1 = [] 


    for pred_loc,target_loc in tqdm(zip(pred_imgs,gt_imgs),total=len(pred_imgs)):
        
        pred_img = cv.imread(pred_loc,cv.IMREAD_GRAYSCALE)
        pred_img = cv.resize(pred_img,(size,size))     
        pred_img = pred_img/255

        pred_foreground = np.zeros((pred_img.shape[0],pred_img.shape[1]))
        pred_background = np.zeros((pred_img.shape[0],pred_img.shape[1]))
        pred_foreground[pred_img > threshold] = 1
        pred_background[pred_img <= threshold]  = 1

        pred_foreground = np.expand_dims(pred_foreground,axis=0)
        pred_background = np.expand_dims(pred_background,axis=0)            
        pred_maps = np.concatenate((pred_background, pred_foreground), axis=0)
        pred_maps = np.expand_dims(pred_maps,axis=0)   
        
        gt_img = cv.imread(target_loc,cv.IMREAD_GRAYSCALE)
        gt_img = gt_img/255
        gt_img = cv.resize(gt_img,(size,size))     

        gt_foreground = np.zeros((gt_img.shape[0],gt_img.shape[1]))
        gt_background = np.zeros((gt_img.shape[0],gt_img.shape[1]))
        gt_foreground[gt_img >= threshold] = 1
        gt_background[gt_img < threshold]  = 1

        gt_foreground = np.expand_dims(gt_foreground,axis=0)
        gt_background = np.expand_dims(gt_background,axis=0) 
        gt_maps = np.concatenate((gt_background, gt_foreground), axis=0)            
        gt_maps = np.expand_dims(gt_maps,axis=0) 
        
        pred_maps = torch.tensor(pred_maps)
        gt_maps = torch.tensor(gt_maps)
        dice = dice_coeff(pred_maps,gt_maps)
        dice = np.squeeze(dice.cpu().numpy())

        hd = hauss_d(pred_maps,gt_maps)

        hd = np.squeeze(hd[0])
        hd = hd

      
        overlap = pred_foreground*gt_foreground # Logical AND
        union = pred_foreground + gt_foreground # Logical OR

        

        model_dhd.append(hd[0][1])
        model_mhd.append(hd[1][1])
        model_dice.append(dice[1])

        pred_fg = pred_foreground.flatten()
        gt_bg = gt_foreground.flatten()
        precision =  precision_score(gt_bg, pred_fg)
        recall =  recall_score(gt_bg, pred_fg)
        f1 = f1_score(gt_bg, pred_fg)

          #iou
        
        IOU = jaccard_score(gt_bg,pred_fg)
        
        model_iou.append(IOU)

        model_prec.append(precision)
        model_recall.append(recall)
        model_f1.append(f1)
        
        
        #model_hd.append()
    
    dhd = np.mean(model_dhd)

    iou = np.mean(model_iou)
   

    mhd = np.mean(model_mhd)


    dce = np.mean(model_dice)


    precism = np.mean(model_prec)
   

    recalm = np.mean(model_recall)


    f1m = np.mean(model_f1)

    
    with open(metrics_txt,"a+") as f: 

        str_to_write = "{} {} {} {} {} {} {} {}\n".format(mod,dhd,mhd,dce,precism,recalm,f1m,iou)
        f.write(str_to_write)

    """

    with open(metrics_txt,"a+") as f:

        for i,model in enumerate(model_preds): 

            f.write("{} {} {} {} {} {} {}\n".format(model,mean_dhd[i],mean_mhd[i],mean_dice[i],mean_prec[i],mean_recall[i],mean_f1[i]))

        f.write("\n")
        best_model =  model_preds[np.argmin(mean_dhd)]
        best_metric =  np.min(mean_dhd)

        f.write("{} {} {}\n".format(best_model,"Best dHD",best_metric))

        best_model =  model_preds[np.argmin(mean_mhd)]
        best_metric =  np.min(mean_mhd)

        f.write("{} {} {}\n".format(best_model,"Best MHD",best_metric))

        best_model =  model_preds[np.argmax(mean_dice)]
        best_metric =  np.max(mean_dice)

        f.write("{} {} {}\n".format(best_model,"Best Dice",best_metric))

        best_model =  model_preds[np.argmax(mean_prec)]
        best_metric =  np.max(mean_prec)

        f.write("{} {} {}\n".format(best_model,"Best Precision",best_metric))

        best_model =  model_preds[np.argmax(mean_recall)]
        best_metric =  np.max(mean_recall)

        f.write("{} {} {}\n".format(best_model,"Best Recall",best_metric))

        best_model =  model_preds[np.argmax(mean_f1)]
        best_metric =  np.max(mean_f1)

        f.write("{} {} {}\n".format(best_model,"Best F1",best_metric))

    """

    

    


    


    
        
        







    


            
            


            


