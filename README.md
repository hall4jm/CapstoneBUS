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

# Methodology

### Data Augmentations

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

### Models
![image](https://user-images.githubusercontent.com/46795053/152703177-3459d9ca-f3d8-46c5-8f63-7f9e197b95f9.png)

### Transfer Learning
Starting from an initial randomized point for our models can result in inconsistent 
training times and performance. Instead, we can choose to use starting parameters for our model 
from pre-trained models. This method known as transfer learning uses the model weights and 
biases from a model trained on a different task and utilizes them as an initial starting point for 
our problem. We began our training with the weights and biases for the models trained 
on the ImageNet dataset.

Each model's final fully connected layer was removed since it was trained to output 
classifications for a separate task. A new fully connected layer with two outputs, known as the 
head of the model for our binary classification problem, was added. Because the initial weights 
for this layer must be randomized, we began training by freezing the entire model, except for this 
layer. Frozen layers of a model cannot be updated during training. By leaving the head unfrozen 
this ensured only the weights and biases for the new classification layer were updated during the 
beginning of training. After updating this layer for the set number of epochs, the entire model 
was unfrozen, and all layers were updated during the rest of the training. 

### Discriminative Learning Rates
The early layers of our model will be used to detect simple features in our images, such 
as edges, curves, and corners. These simple features are present in almost all images. Because 
these layers will begin training with their pre-trained weights, they should require little change. 
However, as we move deeper through the network layers, the model parameters will require 
more change to reach an optimal solution because the complex features found for ImageNet will 
be much different from those for BUS classification. Therefore, it does not seem reasonable that 
all layers should be trained at the same learning rate. We used discriminative learning, which sets 
a different learning rate for different depths of our networks. This ensures that our deeper layers 
and our new head train much faster than the early layers, which should only require minor 
updates. Ultimately, this should reduce our training time as the model learns the features 
important to BUS classification more quickly. 

### Scheduler
It is often the case that the initial learning rate is not optimal throughout all the training. 
For example, we may find a local minimum in our loss function, and our learning rate may not 
allow us to leave this local area of our loss function. For our models, we implement cosine 
annealing, following the 1-cycle policy defined by Smith (2018). Many other schedulers exist, 
but the 1-cycle policy performs well as a universal scheduler and greatly reduces training time. 
The 1-cycle policy works by increasing the learning rate from the starting learning rate to a 
maximum learning rate and then lowering it again to zero over one cycle, typically chosen to be 
slightly less than the total training epochs.