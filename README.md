# Introduction
The aim of this project was to utilize current state-of-the-art CNN architectures for use in a pilot study to examine the use of these algorithms to aid in the classification of breast cancer in breast ultrasounds (BUS) images. The training script included allows for an easy way using fastAI to train and test model architectures.

The script also performs hyperparameter tuning using Tree-structured Parzen Estimator with hyperparameters that can be set in the configuration file passed to the Python script.

The python scripts can be run from the Python console or a Jupyter notebook. 

### Example 

The model and hyperparameter tuning options are stored in a JSON file and passed to the Python console along with the training script.

![image](https://user-images.githubusercontent.com/46795053/145736141-c80eac5c-38f1-47d8-8f82-b6636ceaa6bc.png)

# Background 
According to a recent study, breast cancer now affects more people globally than any 
other form of cancer (Breast cancer now most common form of cancer: WHO taking action, 
2021). It is estimated that over 280,000 new cases will be diagnosed in the United States alone in 
2021, resulting in over 43,000 deaths (U.S. Breast Cancer Statistics, 2021). A critical factor in 
reducing the number of mortalities is ensuring that potentially life-threatening lesions are found 
early (Sun et al., 2017). By finding these lesions early, patients have access to more treatment 
options when they are most effective.

To aid in early detection, doctors recommend regular mammograms or physical 
examinations to help identify abnormalities in breast tissue (Breast Cancer Early Detection and 
Diagnosis, 2021). Another common alternative is the use of ultrasound imaging. Breast 
ultrasounds offer a non-radioactive imaging technique, allowing for a safe and non-invasive way 
to assess any conspicuous masses and limiting the need for unnecessary surgery or invasive 
procedures such as a biopsy.

### BUS Interpretation
Advancements in medical imaging have greatly improved the level of diagnostics that 
can be recorded. These advancements allow radiologists to get a detailed view of breast tissues 
and vascularity, allowing for more accurate diagnoses and better patient care. Despite these 
developments, the interpretations of breast ultrasounds still rely heavily on the experience and 
judgment of radiologists to define characteristics found in the images. Many features described 
in the BI-RAD assessment can be found in benign and malignant lesions. The subjectivity of 
these characteristics can lead to wildly varying assessments across radiologists.

Because BUS images are often a determining factor for the need for a biopsy, low 
sensitivity in identifying malignant lesions results in unnecessary procedures. The user-reliant 
classification of BUS imaging results in non-uniform patient care. Studies show the percentage of positive breast ultrasound biopsies varies as 
much as 51%. For many patients, this means unnecessary invasive procedures and increased 
medical costs. 

Many academic studies have shown the success of deep learning algorithms to 
assist in classifying and detecting disease in medical images. Still, these algorithms have not 
done well to generalize to a clinical setting. This study aims to create a state-of-the-art computer aided diagnosis (CAD) system to create a more standardized approach to assessing BUS images and assigning a diagnosis.

### Significance 
Many existing studies have been performed on publicly available BUS imaging datasets. 
These studies have taken various approaches, including many different machine learning 
algorithms, showing great success on these datasets. However, very few of these systems have 
been implemented in a clinical setting. This CAD system will be unique in its blend of deep 
learning and expert human knowledge trained for a specific patient population. This system, if 
successful, could help to better patient care and reduce medical costs in a clinical setting.

# Methodology

## Data Augmentations

A common problem with deep learning architectures is their ability to memorize instead 
of "learning," making them unable to generalize to new data. With the small amount of training 
data, it became essential to incorporate techniques to reduce overfitting. By randomly selecting 
images during each training step to be altered via rotations, reflections, added noise, Etc., we can 
help the models become more robust. We implemented two different augmentation strategies to 
help avoid overfitting on our BUS images. 

The first augmentation strategy uses geometric transformations, image resizing, and 
zooming. The geometric transformations used on our dataset could not be chosen without some 
considerations. Depending on the transformations applied, the final classification may be altered, 
or the image may no longer make sense in the domain of BUS images. For example, BUS 
images are taken with the layer of skin towards the top and bone or deep tissue towards the 
bottom. If we were to flip the image vertically, we would be introducing a transformation that 
may not be appropriate where the skin was shown beneath bone and dense tissue. Therefore, we 
have limited transformations only to introduce some image rotation by a slight angle, less than 
ten degrees in either direction, zooming the image up to 1.5x magnification, and randomly 
cropping the image to 224×224 pixels

The second data augmentation method utilized was mixup (Zhang et al., 2017). This 
method is a data-agnostic approach to augmentation and does not suffer from relying on domain 
knowledge to apply relevant transformations. According to Zhang et al., this data augmentation 
is also very powerful when the labels are not entirely accurate. This was extremely important for 
our BUS images since many of the images have been classified by a human, which we know are 
prone to errors. 
To perform mixup, we generate a new image by randomly selecting two images, 𝑥௜
, and 
𝑥௝
, and their labels, 𝑦௜
, and 𝑦௝,
 which have been encoded, in our case 0 for benign and 1 for 
malignant. These images and labels are then combined linearly based on a randomly chosen 
weight, λ. Figure 5 shows how the new images, 𝑥ො, and labels, 𝑦, ෝ are generated: 