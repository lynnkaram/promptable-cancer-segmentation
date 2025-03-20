# Promptable cancer segmentation using minimal expert-curated data

Current segmentation models have shown promising results in the medical field, however prostate cancer segmentation specifically remains a challenge. This repository contains codes for a project which integrates two classifiers, one weakly-supervised and one fully-supervised classifier, for segmentating prostate cancer in T2-weighted MRI scans. 

We propose a segmentation method that combines classification and user-defined prompts to localise and delineate regions of interest in volumetric images. A crop of the volumetric image is passed through a weakly-supervised classifier. This predicts the likelihood of object presence in the crop. Simultaneously, a fully-supervised classifier is used to refine classifications of presence by using an edge detection and region overlap mechanism. The crop is then guided by both classifiers in a spiral search, which allows selection of regions where the object is likely present. These regions are then assembled into a pixel-level segmentation mask by combining predictions into a structured output. 

The following figure describes how the framework works. 

![Human interacts with computer](https://github.com/user-attachments/assets/df05960a-722b-47ee-be42-0744f3bf7890)

Use the split_data code to split your data into training, testing and validating datasets. Then, use the latter to train the weakly-supervised object presence classifier, and save the weights. Upload those weights into the click_prompt_app if you want to manually prompt on the image and visualise it, or use the click_prompt_automatic if you want automatic prompting through the dataset. 
