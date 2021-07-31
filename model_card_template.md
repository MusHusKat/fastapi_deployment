# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Developer: Mustafa
* Date: 31st July 2021
* Version: 1
* Type: Random Forest
* Info: training data comes from census data (https://archive.ics.uci.edu/ml/datasets/census* +income). 
* Parameters: Default parameters of scikit-learn's RandomForest Classifier.

## Intended Use
Educational purposes - Udacity Nanodegree Project

## Training Data
Census Data: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
A 30% split from Training Data

## Metrics
* precision
* recall
* fbeta

## Ethical Considerations
Data might be biased. Sliced based metrics have been computed to measure any data and consequently model bias.

## Caveats and Recommendations
No hyperparameter tuning, feature engineering or other optimizations have been performed. The model accuracy is not an end goal in this project.