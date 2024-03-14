# import libraries
import tensorflow as tf
import keras
import numpy as np
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from matplotlib.colors import LinearSegmentedColormap
from tifffile import imsave
#!pip install segmentation-models
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.util import montage
import streamlit as st
import zipfile
from io import BytesIO

# Sidebar
# File uploader
with st.sidebar:
    from io import StringIO
    uploaded_file = st.file_uploader("Upload your folder containing NifTi files", type='zip')
    if uploaded_file is not None:
    # Unzipping the uploaded file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall("users_files")  # Extract to a folder named 'users_files'

# Start by loading model        
st.write('Model is loading...')
from tensorflow.keras.saving import load_model
#Set compile=False as we are not loading it for training, only for prediction.
@st.cache_resource
def load_resnet():
  model1 = load_model('pages/res50_backbone_50epochs_random_50_4.hdf5', compile=False)
  return model1

model1 = load_resnet()
#loading data from the application
@st.cache_data
def load_images():
  data_import_path = 'users_files/import_data/'
  # format requirements
  VOLUME_START_AT = 0
  VOLUME_SLICES = 155
  dim = (224, 224)
  n_channels = 3
  # transform nifti file into array
  X = np.zeros((VOLUME_SLICES, *dim, n_channels))

  for j in range(VOLUME_SLICES):

    temp_image_t2=nib.load(data_import_path + 'import_t2.nii').get_fdata()
    X[j,:,:,0] = temp_image_t2[8:232,8:232,j+VOLUME_START_AT]

    temp_image_t1ce=nib.load(data_import_path + 'import_t1ce.nii').get_fdata()
    X[j,:,:,1] = temp_image_t1ce[8:232,8:232,j+VOLUME_START_AT]

    temp_image_flair=nib.load(data_import_path + 'import_flair.nii').get_fdata()
    X[j,:,:,2] = temp_image_flair[8:232,8:232,j+VOLUME_START_AT]

  ## Loading preprocessing features
  BACKBONE1 = 'resnet50'
  preprocess_input1 = sm.get_preprocessing(BACKBONE1)

  # preprocess input and prediction of segmentation
  X_import = preprocess_input1(X)
  st.write('The model is calculating the prediction for your files.')
  y_pred1=model1.predict(X_import)
  # choose class of segmentation pixels
  y_pred1_argmax=np.argmax(y_pred1, axis=3)

  return X_import, y_pred1, y_pred1_argmax

X_import, y_pred1, y_pred1_argmax = load_images()
# selecting the slice with the largest tumor area by creating a list of all slices
nonzero_count=[]

for i in range(len(y_pred1_argmax)):
  nonzero_count.append(np.count_nonzero(y_pred1_argmax[i,:,:]))
print(nonzero_count)
# select slice with largest tumor area by index
max_index = np.argmax(nonzero_count)

print(max_index)
# loading array of selected slice
test_img_number = max_index
test_img = X_import[test_img_number]

y_pred1_view=y_pred1_argmax[test_img_number,:,:]
test_img_view_t2=test_img[:,:,0]
test_img_view_t1ce=test_img[:,:,1]
test_img_view_t2flair=test_img[:,:,2]

#############################################################

## Create figure of the colored segmentation
colors = [(0, 0, 0),  # black
          (0, 0, 1),  # blue for necrotic core
          (0, 1, 0),  # green for edema
          (1, 1, 0)]  # yellow for enhancing tumor

custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=10)

fig2 = plt.figure(figsize=(6,6))
plt.imshow(y_pred1_view, cmap=custom_cmap)
plt.title('Custom Colormap Image')
plt.axis('off')
fig2.patch.set_facecolor('black')
plt.show()
fig2.savefig('pages/colored_segmentation.png', format='png', bbox_inches='tight')#, pad_iches=0.0)

st.subheader('This is the result of your uploaded files: predicted segmentation of the tumor')
st.write('The image shows the largest tumor area. ')
col1, col2 = st.columns((2,1))
with col1:
   st.image('pages/colored_segmentation.png')
with col2:
   st.write('')
   st.write('')
   st.write('')
   st.write('')
   st.markdown(':blue[necrotic core]')
   st.markdown(':orange[enhancing tumor]')
   st.markdown(':green[edema]')

## Create a figure for each MRI modality and segmentation
fig1, axs = plt.subplots(1, 4, figsize=(20, 5))
fig1.patch.set_facecolor('black')
for ax, data, title in zip(axs, [test_img_view_t2, test_img_view_t1ce, test_img_view_t2flair, y_pred1_view], ['T2', 'T1CE', 'FLAIR', 'Segmentation']):
    if title == 'Segmentation':
      ax.imshow(data, cmap=custom_cmap)
    else:   
      ax.imshow(data, cmap="bone")
    ax.set_title(title, color='white' , fontsize = 20)
    ax.axis('off')  # Remove the axes
    ax.set_facecolor('black')  # Set subplot background to black
plt.subplots_adjust(wspace=0, hspace=0)  # Adjust the space between subplots
plt.show()
fig1.savefig('pages/largest_slice_segmentation.png', format='png', bbox_inches='tight')#, pad_iches=0.0)

st.subheader('Overview of the predicted segmentation within the brain cross-sections')
st.image('pages/largest_slice_segmentation.png')

## Create a figure of all segmentations
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(y_pred1_argmax[:,:,:]), 90, resize=True), cmap=custom_cmap)#cmap ='bone')
ax1.axis('off')
fig.patch.set_facecolor('black')
fig.savefig('pages/custom_segmentation.png', format='png', bbox_inches='tight')#, pad_iches=0.0)
st.subheader('Comprehensive overview of the predicted tumor segmentations over all scans')
st.image('pages/custom_segmentation.png')

st.write('All pictures are AI-generated predictions. No liability is assumed for the results. Please contact your specialist for medical clarification.')