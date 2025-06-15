import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib.RestORI import MMNet
import cv2
import scipy.io as scio
from copy import deepcopy
import time
from PIL import Image
import math

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


    
def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    model = MMNet()
    weights = checkpoint['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return model.eval()

class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.6):
        super(NonMaxSuppression,self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr
        
    def forward(self, repeatability):
        #repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima*0
        border_mask[:,:,10:-10,10:-10]=1
        maxima = maxima*border_mask
        print(maxima.sum())
        return maxima.nonzero().t()[2:4]

import sys
def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract keypoints for a given image")

    parser.add_argument("--num_features", type=int, default=4096, help='Number of features')
    parser.add_argument("--model", type=str, default='Pretrained/VIS_IR.pth', help='model path')
    #parser.add_argument("--model", type=str, default='Pretrained/VIS_SAR.pth', help='model path')
    parser.add_argument("--img1_path", type=str, default='VIRVIS1.png', help='path for VIS image')
    parser.add_argument("--img2_path", type=str, default='VIRIR1.png', help='path for other modal image')
    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--border", type=float, default=5) 
    parser.add_argument("--reliability-thr", type=float, default=0.1)
    parser.add_argument("--repeatability-thr", type=float, default=0.1)
    parser.add_argument("--saveimgPath", type=str, default='Matching_Restult.png')
    parser.add_argument("--gpu", type=int, default=1, help='use -1 for CPU')


    opt = parser.parse_args()


    return opt

def extract_multiscale(net,img1,img2,detector,scale_f=2**0.25, min_scale=0.0, 
                        max_scale=1, min_size=256, 
                        max_size=1024, verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup    
    B, three, H, W = img1.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    X1,X2,Y1,Y2,S1,S2,C1,C2,Q1,Q2,D1,D2 = [],[],[],[],[],[],[],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh,nw =img1.shape[2:]   
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            with torch.no_grad():
                output = net(img1,img2)
                #import pdb;pdb.set_trace()
                descriptors1 = output['feat'][0]
                descriptors2 = output['feat'][1]
                repeatability1 = output['score'][0]
                repeatability2 = output['score'][1]
            mask1=repeatability1*0
            mask2=repeatability2*0
            y1,x1=detector(repeatability1)
            y2,x2=detector(repeatability2)
            q1 = repeatability1[0,0,y1,x1]
            q2 = repeatability2[0,0,y2,x2]
            d1 = descriptors1[0,:,y1,x1].t()
            d2 = descriptors2[0,:,y2,x2].t()
            n1 = d1.shape[0]
            n2 = d2.shape[0]
            X1.append(x1.float()*W/nw)
            X2.append(x2.float()*W/nw)
            Y1.append(y1.float() * H/nh)
            Y2.append(y2.float() * H/nh)
            Q1.append(q1)
            Q2.append(q2)
            D1.append(d1)
            D2.append(d2)
        s /= scale_f
        nh, nw = round(H*s), round(W*s)
        img1 = F.interpolate(img1, (nh,nw), mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, (nh,nw), mode='bilinear', align_corners=False)
        torch.backends.cudnn.benchmark = old_bm

    Y1 = torch.cat(Y1)
    Y2 = torch.cat(Y2)
    X1 = torch.cat(X1)
    X2 = torch.cat(X2)
    #S = torch.cat(S) # scale
    scores1 = torch.cat(Q1) # scores = reliability * repeatability
    scores2 = torch.cat(Q2)
    XYS1 = torch.stack([X1,Y1], dim=-1)
    XYS2 = torch.stack([X2,Y2], dim=-1)
    D1 = torch.cat(D1)
    D2 = torch.cat(D2)      
    return XYS1,XYS2, D1,D2,scores1,scores2
if __name__ == '__main__':
    # import argparse
    import csv
    import os
    args=parse_arguments()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    net = load_network(args.model)
    net = net.cuda()  
    net.eval() 
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)

    img1 = Image.open(args.img1_path).convert('RGB')
    W, H = img1.size
    img11 = TF.to_tensor(img1).unsqueeze(0)
    img11 = (img11-img11.mean(dim=[-1,-2],keepdim=True))/img11.std(dim=[-1,-2],keepdim=True)
    img11 = img11.cuda()

    img2 = Image.open(args.img2_path).convert('RGB')
    W, H = img2.size
    img22 = TF.to_tensor(img2).unsqueeze(0)
    img22 = (img22-img22.mean(dim=[-1,-2],keepdim=True))/img22.std(dim=[-1,-2],keepdim=True)
    img22 = img22.cuda()
    
    xysa,xysb,desca,descb,scoresa,scoresb=extract_multiscale(net,img11,img22,detector,
        scale_f   = args.scale_f, 
        min_scale = args.min_scale, 
        max_scale = args.max_scale,
        min_size  = args.min_size, 
        max_size  = args.max_size, 
        verbose = True)
    
    if len(scoresa)<args.num_features:
        idxs1 = scoresa.topk(len(scoresa))[1]
    else:
        idxs1 = scoresa.topk(args.num_features)[1]
    kp1 = xysa[idxs1].cpu().numpy()
    desc1 = desca[idxs1].cpu().numpy()
    kp_1= [cv2.KeyPoint(point[0], point[1], 1) for point in kp1]
    img_with_keypoints = cv2.drawKeypoints(np.array(img1), kp_1, None)
    threshold = 3

    # 用于存储保留的点
    # filtered_kp1 = []

    # # 遍历 kp1 中的点
    # for i in range(len(kp1)):
    #     keep = True
    #     for j in range(len(filtered_kp1)):
    #         # 计算点 i 和已保留点 j 之间的欧氏距离
    #         distance = np.linalg.norm(kp1[i] - filtered_kp1[j])
    #         if distance < threshold:
    #             keep = False
    #             break
    #     if keep:
    #         filtered_kp1.append(kp1[i])

    # # 转换为 NumPy 数组
    # filtered_kp1 = np.array(filtered_kp1)
    #import pdb;pdb.set_trace()
    if len(scoresb)<args.num_features:
        idxs2 = scoresb.topk(len(scoresb))[1]
    else:
        idxs2 = scoresb.topk(args.num_features)[1]
    kp2 = xysb[idxs2].cpu().numpy()
    desc2 = descb[idxs2].cpu().numpy()
    kp_2= [cv2.KeyPoint(point[0], point[1], 1) for point in kp2]
    img_with_keypoints = cv2.drawKeypoints(np.array(img2), kp_2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []

    for m,n in matches:
        if m.distance < 0.98*n.distance:            
            #print (f"---- {m.distance}, {n.distance}")
            good.append(m)
    sorted_good = sorted(good, key=lambda x: x.distance)

    print( len(sorted_good) )

    # for i in range(len(sorted_good)//20):
    #     print( f"==== {sorted_good[i].distance} ")
    # for mtch in sorted_good:
    #     print( f"==== {mtch.distance} ")


    # 确保每个desc2关键点的唯一匹配
    unique_matches = {}
    for m in good:
        if m.trainIdx not in unique_matches:
            unique_matches[m.trainIdx] = m
        else:
            if m.distance < unique_matches[m.trainIdx].distance:
                unique_matches[m.trainIdx] = m

    # 从unique_matches字典中获取最终的匹配项
    final_good_matches = list(unique_matches.values())

    # 接下来使用final_good_matches来进行后续步骤
    src_pts = np.float32([kp1[m.queryIdx] for m in final_good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in final_good_matches]).reshape(-1,1,2)
    E, mask = cv2.findHomography(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    # E, mask = cv2.findEssentialMat(
    #     src_pts, dst_pts, np.eye(3), threshold=5.0, prob=0.9999,
    #     method=cv2.FM_RANSAC)

    match_mask=mask.ravel().tolist()
    # 计算H矩阵
    inliers_count = np.sum(mask)

    # 将内点的数量写入CSV文件
    csv_file_path = 'TripletDualRep.csv'

    # 检查文件是否存在，以决定是否写入表头
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['inliers_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        if not file_exists:
            writer.writeheader()  # 文件不存在，写入表头
    
        writer.writerow({'inliers_count': inliers_count})
    print("inliers_count:",inliers_count)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = match_mask, # draw only inliers
                    flags = 2)
    
    #import pdb;pdb.set_trace()
    kp1 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1]
    kp2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2]
    img3 = cv2.drawMatches(np.array(img1), kp1, np.array(img2), kp2, final_good_matches, None, **draw_params)
    Image.fromarray(img3).save(args.saveimgPath)
    log_file=open('select.txt','a+')
    log_file.write('epochNum is {}\n'.format(args.model))
    log_file.write('inliers is {} \n'.format(inliers_count))
    log_file.close()
