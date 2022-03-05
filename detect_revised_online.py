from __future__ import division


from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import os,shutil   
import sys
import time
import datetime
import argparse
import pandas as pd


# sys.path.append('/usr/local/lib/python3.7/site-packages')
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.ticker import NullLocator

if __name__ == "__main__":

    

    # parser = argparse.ArgumentParser()
    sleep_time=40
    image_folder="../data_stream"
    model_def="config/yolov3.cfg"
    weights_path="weights/yolov3.weights"
    class_path="data/coco.names"
    conf_thres=0.8
    nms_thres=0.4
    batch_size=10
    n_cpu=0
    img_size=416
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    # opt = parser.parse_args()
    # print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # os.makedirs("output", exist_ok=True)
    

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(class_path)  # Extracts class labels from file



    while 1:
        #开始循环
        kaishi_time=time.time()
        os.makedirs("temp_process",exist_ok=True)
        shutil.rmtree('temp_process')
        os.makedirs("temp_process",exist_ok=True)
        #移动
        process_list = os.listdir(image_folder)
        for f in process_list:
            file_path = os.path.join(image_folder, f)
            shutil.move(file_path,'temp_process')

        dataloader = DataLoader(
            ImageFolder("temp_process", transform= \
                transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_cpu,)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()

        df=pd.DataFrame(columns=('path','car_num','bus_num'))

        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, conf_thres, nms_thres)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            # imgs.extend(img_paths)
            # img_detections.extend(detections)

            # print(img_paths)
            # print(detections)
            for path, detections in zip(img_paths, detections):
                car_num=0
                bus_num=0
                if detections is not None:
                    # Rescale boxes to original image
                    img = np.array(Image.open(path))
                    detections = rescale_boxes(detections, img_size, img.shape[:2])
                    # unique_labels = detections[:, -1].cpu().unique()
                    # n_cls_preds = len(unique_labels)
                    # bbox_colors = random.sample(colors, n_cls_preds)
                    
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                        # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                        if int(cls_pred)==2:
                            car_num=car_num+1
                        if int(cls_pred)==5:
                            bus_num=bus_num+1
                        # if int(cls_pred)==7:
                        #     truck_num=truck_num+1
                        # if int(cls_pred)==3:
                        #     motorbike_num=motorbike_num+1

                # file='./demo_result_online.txt'
                # with open(file, 'a') as f:
                #     f.write(path+',{},{}'.format(car_num,bus_num)+ os.linesep)
                
                df=df.append({'path':path[13:],'car_num':car_num,'bus_num':bus_num},ignore_index=True)

        # print(df.head())
        df['cam']=df.path.str.split('_',expand=True)[[0]]
        # df.datetime=df.datetime.str.slice(0,19)
        # df.datetime=pd.to_datetime(df.datetime,format='%Y-%m-%d-%H-%M-%S')
        df=df.drop(['path'], axis=1)
        df['sum_num']=df['car_num']+df['bus_num']
        df[['sum_num','car_num','bus_num']]=df[['sum_num','car_num','bus_num']].astype('int')
        df_agg_by_mean=df.groupby(['cam']).mean().round(2)
        df_agg_by_mean.to_csv('../result/result_online.csv')
        del df

        end_time=time.time()
        if end_time-start_time<60:
            time.sleep(60-end_time+start_time)
        else:
            print('超时！')

