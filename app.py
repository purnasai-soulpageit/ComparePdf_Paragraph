import streamlit as st
from base64io import Base64IO
import base64
import os

import pdf2image
from PyPDF2 import PdfFileReader
import pdfquery
from model import extract_paragraphs
from difflib import SequenceMatcher
import difflib
from colr import color
import pandas as pd


st. set_page_config(layout="wide")
st.title('PDF Documents Quality Check')
# st.subheader('Upload Any 2 PDFs...')
fileDir = "downloads"

def similar(a, b):
	# A function to compare 2 strings/Paragraphs to return text
    return SequenceMatcher(None, a, b).ratio()

def inline_diff(a, b):
	# A function that can highlight inline differences in 2 paragraphs.
    matcher = difflib.SequenceMatcher(None, a, b)
    def process_tag(tag, i1, i2, j1, j2):
        if tag == 'insert':
            return color('......', fore='black', back='orange')
        elif tag!='equal':
            return color(matcher.a[i1:i2], fore='black', back='orange')
        else:
            return matcher.a[i1:i2]
    return ''.join(process_tag(*t) for t in matcher.get_opcodes())

import torch
import sys
import warnings

torch.cuda.empty_cache()
sys.path.append("unilm")
warnings.filterwarnings("ignore")

from unilm.dit.object_detection.ditod import add_vit_config
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

# Step 1: instantiate config
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("dit/cascade_dit_base.yml")
# thresh
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

matches = []
not_found_paragraphs = []
extracted_data = []
pdf1_bboxes = []
pdf2_bboxes = []
scores = []
pdf1_texts = []
pdf2_texts = []
pdf1_pages = []
pdf2_pages = []

uploaded_files = st.file_uploader('Upload PDFs for comparison', type="pdf",key = "sample", accept_multiple_files = True)

if uploaded_files is not None:
	#st.success('Success message')
	for fileno,pdf_file in enumerate(uploaded_files):	
		# file_details = {"filename":pdf_file.name, "filesize":str(round(pdf_file.size/1000000,2)) +" MB"}
		# st.write("filename: {} (pdf{})".format(pdf_file.name, str(fileno+1)))
		# st.write("filesize: {} MB".format(str(round(pdf_file.size/1000000,2))))
		
		with open(os.path.join(fileDir,pdf_file.name),"wb") as f:
			f.write((pdf_file).getbuffer())
		
		st.info("Extracting Text from pdf{} ({})".format(str(fileno+1),pdf_file.name))

		pdf_file_path= os.path.join(fileDir,pdf_file.name)
		images = pdf2image.convert_from_path(pdf_file_path)
		pdf = PdfFileReader(open(pdf_file_path,'rb'))
		pqpdf = pdfquery.PDFQuery(pdf_file_path)
		extracted_data.append(extract_paragraphs(predictor, classes, pdf_file_path, images, pdf, pqpdf,type = str(fileno)))
		st.write("Done!")


	if extracted_data:
		st.info("Comparing Texts...")
		my_bar = st.progress(0)
		for i, item1 in enumerate(list(extracted_data[0].items())):
			my_bar.progress(int(100*(i/len(extracted_data[0])))+1)
			item1_flag = False
			for item2 in list(extracted_data[1].items()):
				score = similar(item1[1]['text'], item2[1]['text'])
				if score > 0.75:
					pdf1_bboxes.append(int(str(item1[0]).split(".")[1])+1)
					pdf1_pages.append(int(str(item1[0]).split(".")[0])+1)
					pdf2_bboxes.append(int(str(item2[0]).split(".")[1])+1)
					pdf2_pages.append(int(str(item2[0]).split(".")[0])+1)
					scores.append(int(score*100))
					pdf1_texts.append(str(item1[1]['text']))
					pdf2_texts.append(str(item2[1]['text']))
					item1_flag = True

			else:
				if not item1_flag:
					# not_found_paragraphs[item1[0]] = item1[1]['text']
					# not_found_paragraphs.append([item1[0],item1[1]['text']])
					pdf1_bboxes.append(int(str(item1[0]).split(".")[1])+1)
					pdf1_pages.append(int(str(item1[0]).split(".")[0])+1)
					pdf2_bboxes.append(9999)
					pdf2_pages.append(9999)
					scores.append(9999)
					pdf1_texts.append(str(item1[1]['text']))
					pdf2_texts.append('Match Not Found!')

	table_df = pd.DataFrame()
	table_df['Pdf1 Pg no'] = pdf1_pages
	table_df['Pdf1 text no'] = pdf1_bboxes
	table_df['Pdf2 Pg no'] = pdf2_pages
	table_df['Pdf2 text no'] = pdf2_bboxes
	table_df["Text in pdf1"] = pdf1_texts
	table_df["Text in pdf2"] = pdf2_texts
	table_df['Score'] = scores

	for f in os.listdir(fileDir):
		os.remove(os.path.join(fileDir, f))

	# st.balloons()
	st.table(table_df)