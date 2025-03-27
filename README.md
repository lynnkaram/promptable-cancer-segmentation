# Promptable cancer segmentation using minimal expert-curated data

Current segmentation models have shown promising results in the medical field, however prostate cancer segmentation specifically remains a challenge. This repository contains codes for a project which integrates two classifiers, one weakly-supervised and one fully-supervised classifier, for segmentating prostate cancer in T2-weighted MRI scans. 

Here is a segmentation method that combines classification and user-defined prompts to localise and delineate regions of interest in volumetric images. A crop of the volumetric image is passed through a weakly-supervised classifier. This predicts the likelihood of object presence in the crop. Simultaneously, a fully-supervised classifier is used to refine classifications of presence by using an edge detection and region overlap mechanism. The crop is then guided by both classifiers in a spiral search, which allows selection of regions where the object is likely present. These regions are then assembled into a pixel-level segmentation mask by combining predictions into a structured output. 

The following figure describes how the framework works. 


<img width="477" alt="Screenshot 2025-03-20 at 3 20 01 PM" src="https://github.com/user-attachments/assets/ed74db0c-af7d-404e-a7f2-98ebf6bd4e96" />



Use the split_data.py code to split your data into training, testing and validating datasets. Then, use the latter to train the weakly-supervised object presence classifier on the weak_classifier.py code, and save the weights. Upload those weights into the click_prompt_app.py if you want to manually prompt on the image and visualise it, or use the click_prompt_auto.py if you want automatic prompting through the dataset. 
