import streamlit as st
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
from nilearn import image, plotting, datasets, surface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio
from skimage.util import montage
import os


import streamlit as st
import pandas as pd


st.title("Brain Tumor Segmentation")

parag1 = """<div style="text-align: justify;"> Brain tumors can be categorized 
into several types, each with its unique characteristics, origins, and treatment options. 
They can be broadly classified into primary tumors, which originate in the brain,
 and secondary (metastatic) tumors, which have spread to the brain from other parts
   of the body. 
Glioma is a common type of tumor originating in the brain, accounting for approximately
 33 % of all brain tumors.
It starts in the glial cells that support and surround neurons in the brain, including 
astrocytes, oligodendrocytes, and ependymal cells.
Tumor grades can vary from low-grade (less aggressive) to high-grade (more aggressive), 
with glioblastoma being the most aggressive and common form. </div>"""

parag2 = """<div style="text-align: justify;"> The whole tumor describes the union of the 
two sub-regions: the Core and the Edema. The core is consisting of necrotic (non-enhancing)
 and enhancing areas (See Figure on the right). 
The Glioma sub-regions are captured through varying intensity profiles spread across 
multimodal MRI (mMRI) scans,
indicating differences in the biological makeup of the tumor. </div>"""
parag3 = """<div style="text-align: justify;"> The accurate diagnosis and characterization
 of brain tumors are critical for treatment planning and prognosis but are complicated.
Brain tumors can vary greatly in size, shape, location, and contrast enhancement. This 
variability makes it challenging to distinguish between tumor types and to identify tumor
 boundaries accurately.
In response to these difficulties, radiologists are turning more and more to sophisticated 
imaging methods and, notably, to artificial intelligence (AI) technologies aimed at improving
 the accuracy 
and efficiency of diagnoses in MRI scans.
Here, we offer a platform that uses deep learning models to identify tumors and divide them
 into sub-regions by classifying each pixel in each image.  </div>"""

st.header("Brain Tumor")
st.subheader("What is brain tumor?")
st.markdown(parag1,  unsafe_allow_html=True)
st.write('')
st.write('')
st.markdown('''With the technique of Magnetic Resonance Imaging (MRI) the brain 
            is scanned in 3D. Therefore, manyfold pictures of the brain are provided. 
            To get the whole shape of the tumor, the areas are marked in a segmentation file.
            Here the tumor area is marked in blue.''')

#sample_filepath = 'app\data\BraTS20_Training_045\BraTS20_Training_045'
#flair_path = sample_filepath+'_flair.nii'
#t1_path = sample_filepath+'_t1.nii'
#t1ce_path = sample_filepath+'_t1ce.nii'
#t2_path =  sample_filepath+'_t2.nii'
#mask_path = sample_filepath+'_seg.nii'

#sample_img = nib.load(flair_path)
#sample_img = np.asanyarray(sample_img.dataobj)
#sample_mask = nib.load(mask_path)
#sample_mask = np.asanyarray(sample_mask.dataobj)

#image = np.rot90(montage(sample_img))
#mask = np.rot90(montage(sample_mask))
#mask = np.clip(mask, 0, 1)
#label = flair_path.replace('/', '.').split('.')[-2]

#fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
#ax1.imshow(image, cmap ='bone')
#ax1.imshow(np.ma.masked_where(mask == False, mask),
#           cmap='cool', alpha=0.6, animated=True)
#ax1.set_facecolor('black')
#ax1.axis('off')
#fig.savefig(f'{label}_3d_to_2d.png', format='png', bbox_inches='tight')#, pad_iches=0.0)

st.image('data\BraTS20_Training_045\BraTS20_Training_045_flair_3d_to_2d.png', width = 700, use_column_width='never')
st.write('')
st.write('')
st.write('')
col1, col2, col3 = st.columns((1.6, 0.1, 1))
with col1:
    st.subheader("Tumor Sub-regions")

    st.write(parag2 , unsafe_allow_html = True)

with col3:
    st.image("data\sub_regions.png" , caption="Tumor sub-regions", width = 250)

st.write('')
st.write('')
st.subheader('**Multimodal MRI**')
st.write('''Multimodal MRI (Magnetic Resonance Imaging) refers to the use of multiple MRI 
         techniques or sequences in the imaging of tissues, particularly in the brain, 
         to provide a comprehensive evaluation. Each MRI modality captures different aspects 
         of tissue properties, and when combined, they offer a holistic view of the anatomy 
         and pathology.''')

## Plot for different MRT  modalities

#t1_niimg  = nl.image.load_img(t1_path)
#t1de_niimg  = nl.image.load_img(t1de_path)
#t2_niimg  = nl.image.load_img(t2_path)
#flair_niimg = nl.image.load_img(flair_path)

#fig1, axes = plt.subplots(nrows=4, figsize=(30, 20))

#nlplt.plot_epi(t1_niimg, title="MRT T1 modality", axes=axes[0])
#nlplt.plot_epi(t1ce_niimg, title="MRT T1ce modality", axes=axes[1])
#nlplt.plot_epi(t2_niimg, title="MRT T2 modality", axes=axes[2])
#nlplt.plot_epi(flair_niimg, title="MRT flair modality", axes=axes[3])
#plt.show()
images =['T1','T1ce','T2','Flair']

user_choice = st.selectbox('''The four modalities bring out different aspects for the same patient.
                           Choose which MRT modality to display''', options=images)
if user_choice == 'T1':
    st.markdown('***T1 shows the structure and composition of different types of tissue.***')
    st.image('data\BraTS20_Training_045\T1_modality.png', use_column_width=True)

elif user_choice == 'T1ce':
    st.markdown('''***T1ce is similar to T1 images but with the injection of a contrast agent, which will 
             enhance the visibility of abnormalities.***''')
    st.image('data\BraTS20_Training_045\T1ce_modality.png', use_column_width=True)
elif user_choice == 'T2':
    st.markdown('***T2 shows the fluid content of different types of tissue.***')
    st.image('data\BraTS20_Training_045\T2_modality.png', use_column_width=True)

else:
    st.markdown('''***Flair is used to suppress this fluid content, to better identify lesions and tumors 
             that are not clearly visible on T1 or T2 images.***''')
    st.image('data\BraTS20_Training_045\Flair_modality.png', use_column_width=True)


st.write('')

## Plot for segmentation mask

#niimg = nl.image.load_img(flair_path)
#nimask = nl.image.load_img(mask_path)

#colors = [(0, 0, 1),  # blue
#          (0, 1, 0),  # green
#          (1, 1, 0)]  # yellow

#custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=10)
#nlplt.plot_roi(nimask,
#               #title='BraTS20_Training_001_flair.nii with mask plot_roi',
#               bg_img=niimg,
#               cmap=custom_cmap)
    
st.subheader('Tumor segmentation areas')
st.image('data\BraTS20_Training_045\segmentation_mask.png', use_column_width=True)
st.write('')
st.markdown('The glioma is segmented and colored appropriately in three areas: :blue[necrotic core], :orange[enhancing tumor] and :green[edema].')
st.write('')
st.write('')
st.subheader("Challenges")
st.markdown(parag3 , unsafe_allow_html = True)
