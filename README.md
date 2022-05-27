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

Make sure `Cuda 10.2` installed on server

Install Pytorch `torch==1.10.0+cu102` with:

`pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html`


Install Detectron `detectron2==0.6+cu102` with:

`python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html`

#### Command to run app:
`streamlit run app.py --server.fileWatcherType none`