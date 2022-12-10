# Embeddings for Sentiment Analysis

![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

In this work we used the SciKit framework in order to produce different word and sentence embeddings with final objective the optimization of a sentiment analysis classifier.

#### **Disclaimer**

The purpose of the notebook is to show how to apply the various techniques and pipelines. We did not try to find the best model but only tested a number of configurations that would make execution quick.

## The data

The dataset is a list of reviews, containing the text and the sentiment of the review; 1 represents a positive review, while 0 represents a negative review. 

## Used embedding techniques

  * [Tf-idf vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  * [Onehot encoding](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
  * [Transformers](https://www.sbert.net/docs/pretrained_models.html%5D)
  
## Classifiers

  * [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  * [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  * [K-neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
  
## Hyperparameters optimization

  * [Grid Search Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
  * [Grid Search on Validation set](https://www.analyticsvidhya.com/blog/2020/09/alternative-hyperparameter-optimization-technique-you-need-to-know-hyperopt/)

<br />
<br />
<p align="center">
    <img src="https://user-images.githubusercontent.com/50860347/147412786-183da6b0-990f-4016-9f2e-0719d8066f5b.png" style="width: 100%"/>
<p>

<br />
