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
#import cv2
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
            zip_ref.extractall("app/users_files")  # Extract to a folder named 'users_files'
        # Listing all files in the unzipped directory
        #nifti_files = [f for f in os.listdir("app/users_files") if f.endswith('.nii')]
        # Assuming you want to process the first three NIfTI files (if available)
        #selected_files = nifti_files[:3]  # Adjust this logic based on your selection criteria
        #for file in selected_files:
            # Load a NIfTI file
        #    nifti_path = os.path.join("unzipped_files", file)
        #    nifti_img = nib.load(nifti_path)
        # Convert the NIfTI file to a NumPy array
        #img_data = nifti_img.get_fdata()
        # Save the NumPy array as a .npy file
        #npy_path = nifti_path.replace('.nii', '.npy')
        #np.save(npy_path, img_data)
        # Displaying a message
        #st.write(f"{file} has been converted to a NumPy array and saved as {os.path.basename(npy_path)}")
#st.write(uploaded_file)
st.write('Model is loading...')
from tensorflow.keras.saving import load_model
#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('pages/res50_backbone_50epochs_random_50_4.hdf5', compile=False)

#loading data from the application

data_import_path = 'users_files/import_data/'

VOLUME_START_AT = 0
VOLUME_SLICES = 155

dim = (224, 224)
n_channels = 3

X = np.zeros((VOLUME_SLICES, *dim, n_channels))

for j in range(VOLUME_SLICES):

  temp_image_t2=nib.load(data_import_path + 'import_t2.nii').get_fdata()
  X[j,:,:,0] = temp_image_t2[8:232,8:232,j+VOLUME_START_AT]

  temp_image_t1ce=nib.load(data_import_path + 'import_t1ce.nii').get_fdata()
  X[j,:,:,1] = temp_image_t1ce[8:232,8:232,j+VOLUME_START_AT]

  temp_image_flair=nib.load(data_import_path + 'import_flair.nii').get_fdata()
  X[j,:,:,2] = temp_image_flair[8:232,8:232,j+VOLUME_START_AT]


#st.write(X.shape)
#st.write(temp_image_t2.shape)

## Model for prediction
BACKBONE1 = 'resnet50'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_import = preprocess_input1(X)
st.write('The model is calculating the prediction for your files.')
y_pred1=model1.predict(X_import)
y_pred1_argmax=np.argmax(y_pred1, axis=3)

fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(y_pred1_argmax[:,:,:]), 90, resize=True), cmap ='bone')
ax1.axis('off')
fig.patch.set_facecolor('black')
fig.savefig('pages/custom_segmentation.png', format='png', bbox_inches='tight')#, pad_iches=0.0)
st.subheader('AI-assisted segmentated areas for each slice of the scan')
st.image('pages/custom_segmentation.png')

# selecting the slice with the largest tumor area
nonzero_count=[]

for i in range(len(y_pred1_argmax)):

  nonzero_count.append(np.count_nonzero(y_pred1_argmax[i,:,:]))

print(nonzero_count)

max_index = np.argmax(nonzero_count)

print(max_index)


#############################################################


test_img_number = max_index
test_img = X_import[test_img_number]

y_pred1_view=y_pred1_argmax[test_img_number,:,:]
test_img_view_t2=test_img[:,:,0]
test_img_view_t1ce=test_img[:,:,1]
test_img_view_t2flair=test_img[:,:,2]

## Create a figure for each MRI modality and segmentation
fig1, axs = plt.subplots(1, 4, figsize=(20, 5))
fig1.patch.set_facecolor('black')
for ax, data, title in zip(axs, [test_img_view_t2, test_img_view_t1ce, test_img_view_t2flair, y_pred1_view], ['T2', 'T1CE', 'FLAIR', 'Segmentation']):
    ax.imshow(data, cmap="bone")
    ax.set_title(title, color='white' , fontsize = 20)
    ax.axis('off')  # Remove the axes
    ax.set_facecolor('black')  # Set subplot background to black
plt.subplots_adjust(wspace=0, hspace=0)  # Adjust the space between subplots
plt.show()
fig1.savefig('pages/largest_slice_segmentation.png', format='png', bbox_inches='tight')#, pad_iches=0.0)

st.subheader('Crosssections with the largest tumor area')
st.image('pages/largest_slice_segmentation.png')

## Create figure of the colored segmentation
colors = [(0, 0, 0),  # black
          (0, 0, 1),  # blue
          (0, 1, 0),  # green
          (1, 1, 0)]  # yellow

custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=10)

fig2 = plt.figure(figsize=(6,6))
plt.imshow(y_pred1_view, cmap=custom_cmap)
plt.title('Custom Colormap Image')
plt.axis('off')
fig2.patch.set_facecolor('black')
plt.show()
fig2.savefig('pages/colored_segmentation.png', format='png', bbox_inches='tight')#, pad_iches=0.0)

st.subheader('Colored segmentation')
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



#fig2, ax = plt.subplot(1,1,figsize=(6,6))
#ax[0].imshow(y_pred1_view, cmap=custom_cmap)
#ax[0].set_title(title, color='white' , fontsize = 20)
#ax[0].axis('off')  # Remove the axes
#ax[0].set_facecolor('black')

#new_mask_nifti = nib.Nifti1Image(y_pred1, affine=np.eye(4))

#nlplt.plot_roi(new_mask_nifti,
#               #title='BraTS20_Training_001_flair.nii with mask plot_roi',
#               bg_img=temp_image_flair,
#               cmap=custom_cmap)