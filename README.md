
---
## Problem statement

Given a set of longitudinal data of different lab measurements for patients diagnosed with chronic kidney disease (CKD), as well as the information whether these patients progress in their CKD stage or not in the future. 
Using this dataset, predict whether a patient will progress in CKD staging given the patient's past longitudinal information. Will evaluate the models based on recall, to achieve a low rate of type II errors.

---
## Executive Summary

Chronic kidney disease, also called chronic kidney failure, describes the gradual loss of kidney function. Your kidneys filter wastes and excess fluids from your blood, which are then excreted in your urine. When chronic kidney disease reaches an advanced stage, dangerous levels of fluid, electrolytes and wastes can build up in your body.

Chronic kidney disease (CKD) refers to all five stages of kidney damage, from very mild damage in stage 1 to complete kidney failure in stage 5. The stages of kidney disease are based on how well the kidneys can filter waste and extra fluid out of the blood. In the early stages of kidney disease, your kidneys are still able to filter out waste from your blood. In the later stages, your kidneys must work harder to get rid of waste and may stop working altogether.

The way doctors measure how well your kidneys filter waste from your blood is by the estimated glomerular filtration rate, or eGFR. Your eGFR is a number based on your blood test for creatinine, a waste product in your blood

![results](../images/ckd_stages.jpg)
<br><br>

Started off by analyzing the nine data files provided, as shown below:
![metrics](../images/data_files.jpg)

In these files were the information for 300 patients.<br>
Some data processing done were:
1) From each of the lab measurements files, only 2 data points are extracted:
- the most recent value (max time value)
- the linear trend of the values (1: upwards trend, 0: downwards trend)<br>

2) Generate the CKD stage, the MDRD equation to calculate the eGFR was adopted. <br>
GFR=175×(Scr)^-1.154×(Age)^-0.203×(0.742 if female)×(1.212 if African American) <br>
Source: https://github.com/chapmanbe/shilpy_6018_2017_term_project

This dataset has an imbalanced class where the majority class (about 66.7%) shows data for which Stage_Progress is False while about 33.3% of the data shows that Stage_Progress is True. Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class before using the data on our models, as well as class weight were used on selected model.
To determine the overall performance of the model, number of false negatives need to be reduced.

Results from the models are shown in the table below:
![metrics](../images/metrics.jpg)

Based on the the results, we concluded that Logistics Regression with Class Weight has the best performance based on its high recall score. <br>The top five features are 'ldl_trend', 'sbp_trend', 'glu', 'ldl', and 'metoprolol'.

---
### Files used in this projects
```bash
|-- README.md
|-- code
|   |-- CKD.ipynb
|   |-- ckd
|       |-- DataPrep
|           |-- LoadFiles.py
|       |-- Utils
|           |-- Utils.py
|-- data
|   |-- T_creatinine.csv
|   |-- T_DBP.csv
|   |-- T_demo.csv
|   |-- T_glucose.csv
|   |-- T_HGB.csv
|   |-- T_ldl.csv
|   |-- T_meds.csv
|   |-- T_SBP.csv
|   |-- T_stage.csv
|-- images
|   |-- ckd_stages.jpg
|   |-- data_files.jpg
|   |-- metrics.jpg

```
---