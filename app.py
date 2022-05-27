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


st.title('PDF Documents Quality Check')
st.subheader('Upload Any 2 PDFs...')
fileDir = "/home/ubuntu/soulpage/PURNA/pdf_comparision_qc_task/streamlit_demo/downloads/"

def similar(a, b):
	# A function to compare 2 strings/Paragraphs to return text
    return SequenceMatcher(None, a, b).ratio()

def show_pdf(file_path):
	# A function to display the pdf.
	with open(file_path,"rb") as f:
		base64_pdf = base64.b64encode(f.read()).decode('utf-8')
	pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
	st.markdown(pdf_display, unsafe_allow_html=True)

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

uploaded_files = st.file_uploader('Choose your .pdf file', type="pdf",key = "sample", accept_multiple_files = True)
print(uploaded_files)

if uploaded_files is not None:
	#st.success('Success message')
	for fileno,pdf_file in enumerate(uploaded_files):	
		file_details = {"filename":pdf_file.name, "filetype":pdf_file.type,
									"filesize":str(round(pdf_file.size/1000000,2)) +" MB"}
		st.write(file_details)
		
		with open(os.path.join(fileDir,pdf_file.name),"wb") as f:
			f.write((pdf_file).getbuffer())
		
		st.info("Extracting Text from pdf "+str(fileno+1))
		pdf_file_path= os.path.join(fileDir,pdf_file.name)
		images = pdf2image.convert_from_path(pdf_file_path)
		pdf = PdfFileReader(open(pdf_file_path,'rb'))
		pqpdf = pdfquery.PDFQuery(pdf_file_path)
		extracted_data.append(extract_paragraphs(pdf_file_path, images, pdf, pqpdf,type = str(fileno)))
		#output_file =  pdf_file.name.split(".")[-2]+"_Output.pdf"
		#show_pdf(os.path.join(fileDir,output_file))
	
	#download pdf.......
	# for fileno,pdf_file in enumerate(uploaded_files):
	# 	output_file =  pdf_file.name.split(".")[-2]+"_Output.pdf"
	# 	with open(os.path.join(fileDir,output_file)) as fp:
	# 		pdf  = fp.read()
	# 	st.download_button(label="Download PDF", data=pdf, file_name= output_file, mime='application/octet-stream')
	


	if extracted_data:
		for item1 in list(extracted_data[0].items()):
			for item2 in list(extracted_data[1].items()):
				score = similar(item1[1]['text'], item2[1]['text'])
				if score > 0.75:
					pdf1_bboxes.append(int(str(item1[0]).split(".")[1])+1)
					pdf1_pages.append(int(str(item1[0]).split(".")[0])+1)
					pdf2_bboxes.append(int(str(item2[0]).split(".")[1])+1)
					pdf2_pages.append(int(str(item2[0]).split(".")[0])+1)
					scores.append(round(score,2))
					pdf1_texts.append(str(item1[1]['text']))
					pdf2_texts.append(str(item2[1]['text']))
				if score == 0.0:
					# not_found_paragraphs[item1[0]] = item1[1]['text']
					not_found_paragraphs.append([item1[0],item1[1]['text']])

	table_df = pd.DataFrame()
	table_df['Pdf1 Pg no'] = pdf1_pages
	table_df['Pdf1 text no'] = pdf1_bboxes
	table_df['Pdf2 Pg no'] = pdf2_pages
	table_df['Pdf2 text no'] = pdf2_bboxes
	table_df["Text in pdf1"] = pdf1_texts
	table_df["Text in pdf2"] = pdf2_texts
	table_df['Score'] = scores
	st.balloons()
	st.table(table_df)

