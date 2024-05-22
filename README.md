# Project description

This repository centers around the Brain Tumor Segmentation Challenge 2020 ([BraTS2020](http://braintumorsegmentation.org/)). <br>
The data was taken from [kaggle](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/data). <br>
As the folders in this link combine three subsequent years of the same images, this project aimed to model the data of 2020-> see [EDA](EDA_Brain18_19_20.ipynb) research

# Installation instructions
1. Python 3.11.3
2. There is a separate requirement file for each model type. 

# Usage
## Code
The code consists of three parts. There are two different modeling methods as well as the corresponding application. <br>
The first model is U-Net from scratch, which needs a lot of data and takes a lot of time and effort to deliver good results. <br>
The second is the U-Net with ResNet50 backbone, which is already pretrained and therefore much more efficient.
## Some plots

## Link to GoogleDrive with models
The models are stored in Google Drive: 
U-Net from scratch:


[Pretrained model](https://drive.google.com/file/d/17CyD2pbZnwlfre2c4FQNOoOcdhz4HDZy/view?usp=sharing)

Please download and save the model and copy the path into the notebook to load it for prediction.

# Features
- Our highlights -> best dice score
- Here you can find the JEPSis [application](app)
- Here is a detailed [presentation](https://prezi.com/view/GMWldlnVD3CiXib8Z3bC/) of the project results


# [License](LICENSE)

# Acknowledgement
## Contributors
**JEPS**is is an interdisciplinary team of <br>
**J**ennifer Winkler <br>
**E**ugenia Kikrkunow <br>
**P**awel Biedunkiewicz <br>
**S**omayyeh Nemati

## The authors of dataset
[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

## Credits to external libraries or tools
- [Naomi Fridman](https://github.com/naomifridman/Unet_Brain_tumor_segmentation)
- [Dr. Sreenivas Bhattiprolu](https://github.com/bnsreenu/python_for_microscopists)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

# Brain Tumor Segmentation

## Content
This repository centers around the Brain Tumor Segmentation Challenge 2020 (BraTS2020). 
Glioblastomas are frequent brain tumors and show very aggressive characteristics. The 5-year survival rate is limited to 5-7 % (Perelman School of Medicine - University of Pennsylvania). 
This is also related to their extreme intrinsic variability in shape, size and histology.
Therefore a precise AI-assisted diagnostic tool can improve the detection and diagnosis is very high...

## Data
The data was taken from kaggle (https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/data). 
As the folders in this link combine three subsequent years of the same images, this project aimed to model the data of 2020.

## Models
...

## Presentation
https://prezi.com/view/GMWldlnVD3CiXib8Z3bC/

## Requirements
...

## Contributors
**JEPS**is is an interdisciplinary team of <br>
**J**ennifer Winkler <br>
**E**ugenia Kikrkunow <br>
**P**awel Biedunkiewicz <br>
**S**omayyeh Nemati

## Set up the Environment
### **`macOS`** type the following commands : 



- For installing the virtual environment and the required package you can either follow the commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
Or ....
-  use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```

### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


   
## Usage

In order to train the model and store test data in the data folder and the model in models run:

**`Note`**: Make sure your environment is activated.

```bash
python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.


