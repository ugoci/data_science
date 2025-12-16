
# Predicting Art Prices at Auction

Machine Learning Project Brief <br>
December 2025 <br>
by Ugo Ikpeazu <br>

## Contents

1.  Domain Background
2.  Problem Statement
3.  Dataset
    *   Dataset Overview
    *   Feature Engineering Plan
4.  Solution Statement
    *   CatBoost – A Gradient Boost Algorithm
    *   Limitations of CatBoost
5.  Benchmark Model
6.  Evaluation Metrics
7.  Project Design  
    **Bibliography**

***

## 1. Domain Background

The primary domain of this project is Art, this includes paintings, sculptures, installations etc. One of the primary methods for determining the price of a piece of art is through auctions. This project proposes a machine learning model that could be used in different contexts e.g. to support auction preparation for artists and collectors, to assess auction performance etc. The risks of the model are that it could potentially be gamed by Artists who try to find patterns and develop works that align with those patterns, in hopes of securing higher auction prices. This risk already exists in the art industry and is not exacerbated by the implementation of this or similar machine learning models.

***

## 2. Problem Statement

> “No single factor should be considered in isolation or as more important than another,” said Chloe Waddington, partner at London-born gallery Timothy Taylor. “Valuing an artwork is a combination of many factors: institutional recognition, market demand, career stage of the artist, condition, authenticity, medium, et cetera.” (Rabb, 2024)

Considering the domain and situation described above, there are a few problems with the way that art prices are determined (Goodbody, 2025):

*   Famous auctions take place in specific cities and are controlled by legacy institutions.
*   Highly valued pieces tend to be sold within the same channels and networks.
*   Investors (both new and experienced) may not be able to independently verify the value a piece is expected to fetch on the market either before selling or buying.

**Goal:**  
Given a dataset of art prices and features, the goal of this project is to develop a model that can predict future auction sale prices based on specific features. At this stage, the project does not account for the emotional factor associated with art purchases and auctions. Sentiment analysis may be included in future editions. 

***

## 3. Dataset

### 3.1. Dataset Overview

*   **Name:** Art Price Dataset
*   **Description:** Includes 754 artworks sold by Sotheby’s, a leading auction house.
*   **License:** CC BY-NC-SA 4.0
*   **Format:** CSV
*   **Available at:** Kaggle - Art Price Dataset - https://www.kaggle.com/datasets/flkuhm/art-price-dataset 

Features include: image of the piece, price, artist, title, year of creation, position of signature, condition, period, movement.

### 3.2. Feature Engineering Plan

With the exception of images, all features will be included in this first iteration of the project. Future versions will include images to assess how the model evolves when learning on images.

***

## 4. Solution Statement

This project proposes to use **CatBoost** as the primary model.

### 4.1. CatBoost – A Gradient Boost Algorithm

Gradient Boost algorithms work by minimizing the loss function i.e. they work by identifying where things are inaccurate. This is done by creating sequential decision trees that correct the error with each new iteration, until the final model is as close to the target (accurate) as possible. Each new tree corrects the error from the previous tree and not the error of the entire problem. CatBoost improves on this by introducing ordered boosting and ordered target statistics. The algorithm has interfaces including *CatBoost* (a general model), *CatBoostRegressor* (for prediction problems) and *CatBoostClassifier* (for classification problems) (CatBoost, na). 

**How it works:**  
`gradient boosting → ordered boosting → categorical feature handling → regularization`

**Features of CatBoost:**

*   **Ordered Boosting** – GBM algorithms tend to be biased because they may predict on future data points that leak. CatBoost prevents this by using a permutation-driven approach to handle data. 

*   **Ordered target statistics** – CatBoost uses ordered target statistics, meaning that it shuffles the data and performs its tasks on the datapoints from the previous row and not the current row. This allows the algorithm to work as if it were dealing with data in real-time where it wouldn’t have upfront access to future data. In addition, the algorithm ensures that it uses only the models gradients and categorical features, not their target values. In other words, the algorithm looks at the error of the data point(s) and categorical features but does not peek at what the data point will predict. 

*   **Prevents over-fitting** – once the model detects overfitting based on the user-defined validation data, CatBoost can stop the training process. 

*   **Categorical features** – CatBoost works well with non-numerical features. The algorithm makes it possible to set/identify categorical (non-numerical) features which are then integrated to the learning process. To achieve this, the algorithm calculates a statistic for the feature based on previous rows. This ability to work with diverse features also reduces the need for extensive feature engineering since the algorithm can learn from more feature types.

*   **Strong regularization** – regularization penalizes complex models to reduce their variance and improve their ability to generalize on unseen data. CatBoost applies regularization to models.

**Summary:**  
In summary, CatBoost applies the same approach to learning as Gradient Boost in the process of how both algorithms work. Their difference lies in how each algorithm handles the data it uses for training. Unlike regular GBM, CatBoost reduces bias by ensuring that there is no data leakage, meaning that current data is not used in predictions. 

### 4.2. Limitations of CatBoost
Some limitations of CatBoost are:
*   It is slightly slower than of GBM such as LightGBM
*   It is potentially more computationally expensive because of the iterations
*   It may not be relevant or work as well as other algorithms when there are no categorical features 


***

## 5. Benchmark Model

Initially planned benchmark: **Linear Regression** (trained on year and price).  
Updated benchmark: Simplified **CatBoost** using a subset of features (year, price, movement) with no parameter tuning.

***

## 6. Evaluation Metrics

*   **R²:** Explains how much variation in the target variable is explained by the model.
*   **Mean Absolute Error (MAE):** Measures the average size of prediction errors.

***

## 7. Project Design

Steps and logic:

*   Data identification and sourcing
*   Exploratory Data Analysis (EDA)
*   Project setup: module imports, dataset initialization
*   Handle missing values
*   Assign X and y labels for benchmark
*   Split data (80-20) for benchmark
*   Fit benchmark model
*   Assign X and y labels for primary model
*   Split data (80-20) for primary model
*   Initialize CatBoost
*   Define categorical features
*   Fit CatBoost model
*   Test model (predict on test data)
*   Evaluate models
*   Document findings

***

## Bibliography
*   CatBoost. (na). Training. From CatBoost: https://catboost.ai/docs/en/features/training
*   Goodbody, L. (2025, July 17). How Auction Houses Determine Art Valuations. From My Art Broker: https://www.myartbroker.com/auction/articles/how-auction-houses-value-art
*   Rabb, M. (2024, November 13). What Determines the Price of an Artwork? From Artsy: https://www.artsy.net/article/artsy-editorial-determines-price-artwork


***

