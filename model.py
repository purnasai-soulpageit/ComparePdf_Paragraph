import os
# os.system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html')
# os.system("git clone https://github.com/microsoft/unilm.git")
# import sys
import math
# import warnings
import torch
import streamlit as st
import numpy as np
# from PIL import Image

# torch.cuda.empty_cache()
# sys.path.append("unilm")
# warnings.filterwarnings("ignore")

# from unilm.dit.object_detection.ditod import add_vit_config
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import ColorMode, Visualizer
# from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultPredictor

# # Step 1: instantiate config
# cfg = get_cfg()
# add_vit_config(cfg)
# cfg.merge_from_file("dit/cascade_dit_base.yml")
# # thresh
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# # Step 2: add model weights URL to config
# cfg.MODEL.WEIGHTS = "https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth"

# # Step 3: set device
# cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Step 4: define model
# predictor = DefaultPredictor(cfg)

# md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
# classes  =["text","title","list","table","figure"]
# md.set(thing_classes= classes)

def analyze_image_box(img,predictor,classes):
    
    """
    A function to take image as input and returns
    image with bboxes on it, all bboxes,scores.
    """
    output = predictor(img)["instances"]
    boxes = output.pred_boxes.to("cpu")
    scores = output.scores.to("cpu")
    final_classes = [classes[label] for label in output.pred_classes.to("cpu")]

    # v = Visualizer(img[:, :, ::-1],
    #                 md,
    #                 scale=1.0,
    #                 instance_mode=ColorMode.SEGMENTATION)
    
    final_list = []
    for box,label,score in zip(boxes, final_classes, scores):
        box = box.tolist()
        final_list.append([box[0],box[1],box[2],box[3], label, round(score.item(),2)])
    final_list = sorted(final_list, key = lambda x:x[1])

    final_boxes = []
    # final_scores = []
    for box0,box1,box2,box3,label,score in final_list:
        if label == "text" or "title":
            box = [box0, box1, box2, box3]
            # v.draw_box(box)
            final_boxes.append(box)
            # v.draw_text(str(label + str(round(score,2))) , box[:2])
            # final_scores.append(round(score,2))
    
    # v = v.get_output()
    # result_image =  v.get_image()[:, :, ::-1]
    return final_boxes

def convert_imgbbox_to_pdfbbox(box, img, page):

    """A function to convert Image bboxes to PDF bboxes"""
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

# @st.cache
def extract_paragraphs(predictor, classes, pdf_file, images, pdf, pqpdf,type = "None"):

    """A function to extract paragraph text using images, to extract bboxes,
    and using those bboxes to extract text"""
    data = {}
    # output_images = []
    # output_file =  pdf_file.split(".")[-2]+"_Output.pdf"

    my_bar = st.progress(0)

    for pg_no, img in enumerate(images):

        my_bar.progress(int(pg_no/len(images)*100)+1)
        # st.progress(pg_no/len(images))

        img = np.array(img)
        with torch.no_grad():
            all_boxes = analyze_image_box(img,predictor, classes)

        print("PDF no:",type,"Page no :", pg_no)

        for box_no,box in enumerate(all_boxes):
            new_box = convert_imgbbox_to_pdfbbox(box, img, pdf.getPage(pg_no))
            tbox = [math.ceil(b) for b in new_box]
            
            co = ','.join([str(int(cord)) for cord in tbox])
            query = f'LTTextLineHorizontal:overlaps_bbox("{co}")'
            
            print("box no :",box_no)
            pqpdf.load(pg_no)
            text = pqpdf.pq(query).text().replace("- ","")
            item = {"pgno":pg_no, "boxno":box_no,"box":tbox,"text":text}
            data[str(pg_no)+"."+str(box_no)] = item

        
        # output_images.append(Image.fromarray(result_image))
    
    torch.cuda.empty_cache()
    
    # output_images[0].save(output_file, save_all=True, append_images= output_images[1:])
    return data

