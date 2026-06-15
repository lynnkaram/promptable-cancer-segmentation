# Citation

Please cite as below: 

Lynn Karam, Yipei Wang, Veeru Kasivisvanathan, Mirabela Rusu, Yipeng Hu, and Shaheer U. Saeed. Promptable Cancer Segmentation Using Minimal Expert-Curated Data. Medical Image Understanding and Analysis (MIUA), 2025.
DOI: https://doi.org/10.48550/arXiv.2505.17915

# Promptable cancer segmentation using minimal expert-curated data

Current segmentation models have shown promising results in the medical field, however prostate cancer segmentation specifically remains a challenge. This repository contains codes for a project which integrates two classifiers, one weakly-supervised and one fully-supervised classifier, for segmentating prostate cancer in T2-weighted MRI scans. 

Here is a segmentation method that combines classification and user-defined prompts to localise and delineate regions of interest in volumetric images. A crop of the volumetric image is passed through a weakly-supervised classifier. This predicts the likelihood of object presence in the crop. Simultaneously, a fully-supervised classifier is used to refine classifications of presence by using an edge detection and region overlap mechanism. The crop is then guided by both classifiers in a spiral search, which allows selection of regions where the object is likely present. These regions are then assembled into a pixel-level segmentation mask by combining predictions into a structured output. 

The following figure describes how the framework works. 

<img width="732" height="507" alt="Screenshot 2026-06-15 at 3 57 43 PM" src="https://github.com/user-attachments/assets/bcda69c3-f452-4808-9d47-c23264212760" />

<img width="697" height="327" alt="Screenshot 2026-06-15 at 3 58 43 PM" src="https://github.com/user-attachments/assets/cfbe7075-4197-4acc-9a50-080073b5c586" />


Use the split_data.py code to split your data into training, testing and validating datasets. Then, use the latter to train the weakly-supervised object presence classifier on the weak_classifier.py code, and save the weights. Upload those weights into the click_prompt_app.py if you want to manually prompt on the image and visualise it, or use the click_prompt_auto.py if you want automatic prompting through the dataset. 
