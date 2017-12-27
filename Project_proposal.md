#  **1. Automating breast cancer diagnosis using histological data (ICIAR2018 Challenge)**
url: https://iciar2018-challenge.grand-challenge.org
***

- Breast cancer diagnosis using microscopic images.
- Build detection and evaluation tool to reduce human evaluation time of histological images.

**Dataset 1**
- H&E stained breast histology microscopy images (.tiff)
- RGB (2048x1536)
- 400+ labeled microscopy images (4x classes: 100 samples per class)

**Benign example:**

![Benign](https://iciar2018-challenge.grand-challenge.org/site/ICIAR2018-Challenge/serve/public_html/benign.png/)

**Dataset 2**
- 10 pixel-wise labeled and 20 non-labeled (.svs and .xml annotations)
- RGB (variable image size)

**Example:**

![](https://iciar2018-challenge.grand-challenge.org/site/ICIAR2018-Challenge/serve/public_html/A08_thumb.png/)

**Part A** - Classification task using dataset 1
- Four classes: normal, benign, in situ carcinoma and invasive carcinoma

**Part B** - Detection task using dataset 2
- Pixel-wise labelling task



Independent test dataset will be provided on 23th January 2018
Python script is provided for reading .svs and the .xml
Part of the ICIAR 2018 conference

Reference paper: 
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544


***

#  **2. Automating breast cancer diagnosis using histological data (CAMELYON17)**
url: https://camelyon17.grand-challenge.org/home/
***

- Automated detection and classification of breast cancer metastases in whole-slide images of histological lymph node sections

- Last year’s CAMELYON16 focused on the detection of lymph node metastases, both on a lesion-level and on a slide-level. This year we move up to patient-level analysis, which requires combining the detection and classification of metastases in multiple lymph node slides into one outcome: a pN-stage
N-stage - whether the cancer has spread to the regional lymph nodes

**Dataset**

- whole-slide images (WSI) of hematoxylin and eosin (H&E) stained lymph node sections.
On a lesion-level: with detailed annotations of metastases in WSI.
On a patient-level: with a pN-stage label per patient.
-  For training, 100 patients will be provided and another 100 patients for testing. This means we will release 1000 slides with 5 slides per patient .

**Example:**

![](https://camelyon17.grand-challenge.org/site/CAMELYON17/serve/public_html/example_high_resolution.png/)


***

# **3. Diabetic retinopathy (DR) detection using retinal fundus images**
- development and evaluation of image analysis algorithms for early detection of diabetic retinopathy. 
- Several databases exists with OCT data for DR detection.
- The idea is to build a DR classifier by merging two datasets. The idea is take advantage of the existent of large dataset (dataset 2 described below) to train a model with this data and evaluate it with a smaller, but more reliable dataset (dataset 1 described below).

**Dataset 1 - IDRiD (Indian Diabetic Retinopathy Image Dataset (Challenge)**
https://idrid.grand-challenge.org/data/
- The first database representative of an Indian population.
- The only dataset constituting typical diabetic retinopathy lesions and normal retinal structures annotated at a pixel level
- Has Information on the disease severity of diabetic retinopathy, and diabetic macular edema for each image

- 516 images
- RGB (4288x2848 in .jpg format)
- 50-degree fov, centered near to the macula
-  81 (DR) and 164 (non-DR)
- Pixel level annotations - Micro aneurysms, soft exudates, hard exudates and haemorrhages
- fov annotations 
- Gradings on 516 images (grading based on Clinical Diabetic Retinophaty Scale - .csv)
Classification scale: http://www.icoph.org/downloads/Diabetic-Retinopathy-Scale.pdf
The macular edema severity was decided based on occurrences of hard exudates near to macula center region.
- Availability of Training Set: January End
- Availability of Testing Set: Early March


**Example:**

![](https://idrid.grand-challenge.org/site/IDRiD/serve/public_html/sample.jpg/)

**Challenge tasks**

**Task 1** - Segmentation of retinal lesions associated with diabetic retinopathy as microaneurysms, hemorrhages, hard exudates and soft exudates
**Task 2** - Disease Grading: Classification of fundus images according to the severity level of diabetic retinopathy and diabetic macular edema. 
**Task 3** - Optic Disc and Fovea Detection: Automatic localization of optic disc and fovea center coordinates and also segmentation of optic disc.
- This Challenge is organized in conjunction with the 2018 IEEE International Symposium on Biomedical Imaging  (ISBI-2018).


# **Dataset 2** - Diabetic retinopathy detection (Kaggle competition)
https://www.kaggle.com/c/diabetic-retinopathy-detection


- 35,000 images
- left and right is provided for every subject
- A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR


- Data is more corrupted:

"The images in the dataset come from different models and types of cameras, which can affect the visual appearance of left vs. right. Some images are shown as one would see the retina anatomically (macula on the left, optic nerve on the right for the right eye). Others are shown as one would see through a microscope condensing lens (i.e. inverted, as one sees in a typical live eye exam)."


***
