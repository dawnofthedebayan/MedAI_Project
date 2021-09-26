import glob 
from metrics import  Dice_Coefficient, All_Hausdorff_Distances_Binary_Image
import os
import cv2 as cv
import numpy as np
import torch
from sklearn.metrics import precision_score,f1_score,recall_score
from tqdm import tqdm

result_path = "/home/debayan/Desktop/MedAI_Project/results/results_cvcclinic.txt"

#models = ["hardnet_kvseg","hardnetours_kvseg","hardnet_etis","hardnetours_etis","hardnet_cvc_colon","hardnetours_cvc_colon","hardnet_cvc_clinic","hardnetours_cvc_clinic"]
#models = ["resunet_kvseg","unet_kvseg"]
#models = ["hardnetablation_onlyse","hardnetablation_rr_2","hardnetablation_rr_8"]
#models = ["AttnUNetnormal","unetnormal","resunet_new_arch_tversky","r2unetnormal","unet_"]
models = ["UNet_","SegNet_","PraNet_","RAResidualFuseDecoder_","ResidualFuseDecoder_"]

datasets = ["cvcclinic"]


for mod in models:

    for ds in datasets: 
        
        dice = []
        mhd = []
        dhd = []
        prec = []
        rec = []
        f1 = []
        iou = []
        
        with open(result_path,"r") as f: 
            
            lines = f.readlines() 
            for line in lines: 
                print(line)
                if ds in line and mod in line:
                    print("INSIDE",ds,mod)

                    line_arr = line.split()
                    dhd.append(float(line_arr[1]))
                    mhd.append(float(line_arr[2]))
                    dice.append(float(line_arr[3]))
                    prec.append(float(line_arr[4]))
                    rec.append(float(line_arr[5]))
                    f1.append(float(line_arr[6]))
                    iou.append(float(line_arr[7]))

    

        if mod == "sq1dTsq2dTsFsaFrr4":
            
            dhd  = sorted(dhd, reverse=False)
            dhd = dhd[:]

            mhd  = sorted(mhd, reverse=False)
            mhd = mhd[:]

            dice  = sorted(dice, reverse=False)
            dice = dice[:]

            prec  = sorted(prec, reverse=False)
            prec = prec[:]

            rec  = sorted(rec, reverse=False)
            rec = rec[:]

            f1  = sorted(f1, reverse=False)
            f1 = f1[:]

            

        else:
            dhd  = sorted(dhd, reverse=False)
            dhd = dhd[:]

            mhd  = sorted(mhd, reverse=False)
            mhd = mhd[:]

            dice  = sorted(dice, reverse=False)
            dice = dice[:]

            prec  = sorted(prec, reverse=False)
            prec = prec[:]

            rec  = sorted(rec, reverse=False)
            rec = rec[:]

            f1  = sorted(f1, reverse=False)
            f1 = f1[:]

            iou  = sorted(iou, reverse=False)
            iou = iou[:]




        with open("/home/debayan/Desktop/MedAI_Project/results/consolidated_results.txt","a+") as f: 

            f.write("{} {}\nDHD:{}/{}\nMHD:{}/{}\nDICE:{}/{}\nPrecision:{}/{}\nRecall:{}/{}\nF1:{}/{}\nmIOU:{}/{}\n\n\n".format(mod,ds,np.mean(dhd),np.std(dhd),np.mean(mhd),np.std(mhd),np.mean(dice),\
                                                                        np.std(dice),np.mean(prec),np.std(prec),np.mean(rec),np.std(rec),np.mean(f1),np.std(f1),np.mean(iou),np.std(iou)))


                

                



"""
meanDic:0.915;meanIoU:0.859;wFm:0.914;Sm:0.942;meanEm:0.978;MAE:0.008;maxEm:0.987;maxDice:0.920;maxIoU:0.866;meanSen:0.912;maxSen:1.000;meanSpe:0.978;maxSpe:0.986.
meanDic:0.911;meanIoU:0.854;wFm:0.909;Sm:0.941;meanEm:0.976;MAE:0.009;maxEm:0.984;maxDice:0.918;maxIoU:0.862;meanSen:0.913;maxSen:1.000;meanSpe:0.977;maxSpe:0.986.


meanDic:0.938;meanIoU:0.891;wFm:0.932;Sm:0.951;meanEm:0.983;MAE:0.010;maxEm:0.989;maxDice:0.945;maxIoU:0.900;meanSen:0.950;maxSen:1.000;meanSpe:0.990;maxSpe:0.999.            
            
meanDic:0.940;meanIoU:0.892;wFm:0.934;Sm:0.954;meanEm:0.984;MAE:0.010;maxEm:0.989;maxDice:0.947;maxIoU:0.901;meanSen:0.950;maxSen:1.000;meanSpe:0.989;maxSpe:0.998.
"""
            


