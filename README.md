# MSCI 436 (S2022): Regression for World Happiness Score

> **Author: Team 9**  
> Vatsal Bora, Shivani Chidella, Soohyun Kim, Louis Liu  
> *(Note: We referenced [MoemenGaafar's repository](https://github.com/MoemenGaafar/World-Happiness-Machine-Learning/blob/main/Project_Notebook.ipynb))*

## Introduction

Hi, we are Team 9; and this is a repository for our codebase.  

Our primary objective is **to help people** with an interest in subjective well-being (including politicians and bureaucrats) **understand which feature impacts the Happiness Score the most** so that they can try improving the policies accordingly.  

To gain some understanding of our project before playing around with our Decision Support System, please refer to the documents below:  
* [ML Canvas for DSS](https://docs.google.com/document/d/1u4ShCuBDY856Qx5RbafQVW9qb2eHTPUJ/edit?usp=sharing&ouid=104657686272436864334&rtpof=true&sd=true)  
* [Status Report (for development)](https://docs.google.com/document/d/1HikO9uLtq5pj5dgth8HtPZjJlEASzyajwIKiX_4ZXDM/edit?usp=sharing)  
* [Presentation Video](https://youtu.be/nHoNV7WtXsc)  
* [Presentation Slide Deck](https://docs.google.com/presentation/d/1rCmUoYg46iJuAbulOO3vN4OJYQSnN1bjTHwtLgRquO0/edit?usp=sharing)  
* [Demo of User Interface](https://youtu.be/tT1yHCS_4_Y)

## Data

We used the ["World Happiness Report 2020"](https://www.kaggle.com/datasets/londeen/world-happiness-report-2020) dataset from Kaggle.

## Model

To achieve our main objective, we decided to create a regression model on the chosen dataset.  

This way, we would be able to determine the best predictor of the Happiness Score and predict the score given the value of each feature.  

As mentioned earlier, we referenced [MoemenGaafar's repository](https://github.com/MoemenGaafar/World-Happiness-Machine-Learning/blob/main/Project_Notebook.ipynb) to build a model.

## User Interface

We used Streamlit to make our own user interface.  

![Snapshot 1](./user-interface-images/snapshot-of-user-interface-1.PNG)
![Snapshot 2](./user-interface-images/snapshot-of-user-interface-2.PNG)
![Snapshot 3](./user-interface-images/snapshot-of-user-interface-3.PNG)

## Instructions

1. If you would like to **run the code on your local machine**, copy-paste the following command in the terminal:

    ```
    git clone https://github.com/s00hyunkim/world-happiness-regression.git
    ```

2. Make sure that you **install all the required libraries** before running any of the code files.

3. If you would like to **run the regression models**, copy-paste the following command in the terminal:

    ```
    python world-happiness-regression.py
    ```

4. If you would like to **directly interact with our user interface**, copy-paste the following command in the terminal:

    ```
    streamlit run world-happiness-user-interface.py
    ```
