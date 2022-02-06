# Introduction
The aim of this project was to utilize current state-of-the-art CNN architectures for use in a pilot study to examine the use of these algorithms to aid in the classification of breast cancer in breast ultrasounds (BUS) images. The training script included allows for an easy way using fastAI to train and test model architectures.

The script also performs hyperparameter tuning using Tree-structured Parzen Estimator with hyperparameters that can be set in the configuration file passed to the Python script.

The python scripts can be run from the Python console or a Jupyter notebook. 

## Example 

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

## BUS Interpretation
The Breast Imaging Reporting and Data System (BI-RADS) provide a standardized way 
for radiologists to describe breast lesions found in BUS images. This standardization aims to 
improve the quality of the assessment, improving patient care. The system defines seven levels 
of ranking 0 – 6, where a higher score reflects an increased likelihood of malignancy. The 
probability of malignancy for each score is shown in Table 1 below (Mendelson et al., 2013): 
Table 1 
BI-RADS Score and Probability of Malignancy 
Score Classification Probability of Malignancy 
0 Incomplete N/A 
1 Negative 0% 
2 Benign 0% 
3 Probably Benign <2% 
4 Suspicious for Malignancy 2-94% 
5 Highly Suggestive of Malignancy >95% 
6 Known Malignancy 100% 
