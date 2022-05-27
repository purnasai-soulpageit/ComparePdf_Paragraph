# ComparePdf_Paragraph

### File structure.
```
├── dit
├── downloads
├── samples
└── app.py
└── model.py
```

#### File usage:
- dit = has Deeplearning model Config files.
- downloads = output pdfs get stored to this folder.
- samples = have sample folders to upload and for testing.
- app.py= Main file that runs streamlit properties and calls Deeplearning model methods.
- model.py = Has Deeplearning methods and Image parsing, model defintion and inference.

`pip install -r requirements.txt`

`torch==1.10.0+cu102`

`detectron2==0.6+cu102`

#### Command to run:
`streamlit run app.py`


