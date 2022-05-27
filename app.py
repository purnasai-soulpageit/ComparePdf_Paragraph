import os
# os.system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html')
# os.system("git clone https://github.com/microsoft/unilm.git")
import streamlit as st
import cv2
import pdf2image
import numpy as np
import matplotlib.pyplot as plt

import math
import json
from PIL import Image
import pdfquery
from PyPDF2 import PdfFileReader

import sys
sys.path.append("unilm")

import cv2
import warnings
warnings.filterwarnings("ignore")

from unilm.dit.object_detection.ditod import add_vit_config

import torch
torch.cuda.empty_cache()

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

# import gradio as gr

# Step 1: instantiate config
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("dit/cascade_dit_base.yml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Step 2: add model weights URL to config
cfg.MODEL.WEIGHTS = "https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth"

# Step 3: set device
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Step 4: define model
predictor = DefaultPredictor(cfg)

md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
classes  =["text","title","list","table","figure"]
md.set(thing_classes= classes)

def analyze_image_box(img):
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    output = predictor(img)["instances"]
    boxes = output.pred_boxes.to("cpu")
    scores = output.scores.to("cpu")
    final_classes = [classes[label] for label in output.pred_classes.to("cpu")]

    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    
    final_list = []
    for box,label,score in zip(boxes, final_classes, scores):
        box = box.tolist()
        final_list.append([box[0],box[1],box[2],box[3], label, round(score.item(),2)])
    final_list = sorted(final_list, key = lambda x:x[1])

    final_boxes = []
    final_scores = []
    for box0,box1,box2,box3,label,score in final_list:
        if label == "text" or "title":
            box = [box0, box1, box2, box3]
            v.draw_box(box)
            final_boxes.append(box)
            v.draw_text(str(label + str(round(score,2))) , box[:2])
            final_scores.append(round(score,2))
    
    v = v.get_output()
    result_image =  v.get_image()[:, :, ::-1]
    return result_image, final_boxes, final_scores

def convert_imgbbox_to_pdfbbox(box, img, page):
    img_height, img_width, _ = img.shape
    _,_,page_width, page_height = page.mediaBox
    x,y,x1,y1 = box

    newx = x*(page_width/ img_width)
    newy = y*(page_height/ img_height)
    newx1 = x1*(page_width/ img_width)
    newy1 = y1*(page_height/ img_height)

    newy  =  page_height - newy
    newy1 =  page_height - newy1
    return [newx, newy1, newx1, newy]

@st.cache
def extract_paragraphs(pdf_file, images, pdf, pqpdf,type = "None"):
    data = {}
    output_images = []
    output_file =  pdf_file.split(".")[-2]+"_Output.pdf"

    for pg_no, img in enumerate(images):
        img = np.array(img)
        result_image, all_boxes, all_scores = analyze_image_box(img)
        for box_no,box in enumerate(all_boxes):
            new_box = convert_imgbbox_to_pdfbbox(box, img, pdf.getPage(pg_no))
            tbox = [math.ceil(b) for b in new_box]
            
            co = ','.join([str(int(cord)) for cord in tbox])
            query = f'LTTextLineHorizontal:overlaps_bbox("{co}")'
            
            print(type, "page no: ", pg_no, "box no :",box_no)
            pqpdf.load(pg_no)
            text = pqpdf.pq(query).text().replace("- ","")
            item = {"pgno":pg_no, "boxno":box_no,"box":tbox,"text":text}
            data[str(pg_no)+"."+str(box_no)] = item
        print("pdf no:",type,"Page no :", pg_no)
        output_images.append(Image.fromarray(result_image))

    output_images[0].save(output_file, save_all=True, append_images= output_images[1:])
    return data


