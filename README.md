# Automatic stenosis recognition from coronary angiography using convolutional neural networks (CMPB 2020 published.)

This repository provides the beta version code for ("Automatic stenosis recognition from coronary angiography(CAG)")[https://doi.org/10.1016/j.cmpb.2020.105819 
].



Abstract

 Background and objective: Coronary artery disease is a leading cause of death and is mostly caused by atherosclerotic narrowing of the coronary artery lumen. Coronary angiography is the standard method to estimate the severity of coronary artery stenosis, but is frequently limited by intra- and inter-observer variations. We propose a deep-learning algorithm that automatically recognizes stenosis in coronary angiographic images.

Methods: The proposed method consists of key frame detection, deep learning model training for classification of stenosis on each key frame, and visualization of the possible location of the stenosis. Firstly, we propose an algorithm that automatically extracts key frames essential for diagnosis from 452 right coronary artery angiography movie clips. Our deep learning model is then trained with image-level annotations to classify the over 50 % narrowed areas. To make the model focus on the salient features, we applied a self-attention mechanism. The stenotic locations were visualized using the activated area of feature maps with gradient-weighted class activation mapping.

Results: The automatically detected key frame was very close to the manually selected key frame (average distance 1.70 ± 0.12 frame per clip). The model was trained with key frames on internal datasets, and validated with internal and external datasets. Our training method achieved high frame-wise area under the curve of 0.971, frame-wise accuracy of 0.934, and clip-wise accuracy of 0.965 in the average values of cross-validation evaluations. The external validation results showed high performances with the mean frame-wise area under the curve of (0.925 and 0.956) in the single and ensemble model, respectively. Heat map visualization shows the location for different types of stenosis in both internal and external data sets. With the self-attention mechanism, the stenosis could be localized precisely and helps to accurately classify stenosis by type of stenosis.

Conclusions: Our automated classification algorithm could recognize and localize coronary artery stenosis highly accurately. Our approach might be the basis of screening and assistant tool for interpretation of coronary angiography.



Keywords

Coronary angiography; Coronary artery stenosis; Deep learning; Stenosis recognition; Automated screening



Original clip         |  Recognized stenosis clip
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/47732974/152632085-58e07c15-0123-45ff-89e5-aeb1af41b170.gif" alt="drawing" width="400"/>  |  <img src="https://user-images.githubusercontent.com/47732974/152632097-de7c6859-14dd-477e-be6b-4942efe8136c.gif" alt="drawing" width="400"/>



