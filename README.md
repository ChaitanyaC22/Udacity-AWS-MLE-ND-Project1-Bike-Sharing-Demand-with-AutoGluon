# Predict Bike Sharing Demand with AutoGluon

## Introduction to AWS Machine Learning Final Project
This project focuses on using AWS open-source AutoML library, AutoGluon to predict the bike sharing demand using the Kaggle Bike Sharing demand dataset. The ulimate objective is to use AutoGluon's 'Tabular Prediction' to achieve accurate AutoML-based baseline models without dealing with a lot of cumbersome issues like data cleaning, feature engineering, hyperparameter optimzation, model selection, etc.

### Source: AWS Machine Learning Engineer Nanodegree Scholarship Program

## Overview
This project was a part of the project assessment in the **'AWS x Udacity's Machine Learning Engineer Nanodegree Scholarship Program'.**

## Problem Overview:

Bicycle sharing programs, also known as bike-sharing systems, offer a convenient and automated method for renting bicycles through a network of kiosk locations dispersed throughout urban areas. These systems enable individuals to rent a bike from one location and return it to another on an as-needed basis. At present, there are over 500 bike-sharing programs in operation worldwide.

The vast amount of data generated by these systems makes them an attractive subject for researchers as they provide explicit records of travel duration, departure and arrival location, and time elapsed. As such, bike-sharing systems serve as a sensor network, providing valuable insights into the mobility patterns of a city. This competition invites participants to utilize historical usage patterns and weather data to predict bike rental demand in the Washington D.C. based Capital Bikeshare program.

## Dataset Link:
![Bike-Sharing-Demand-Image](img/Bike-Sharing-Demand-img.jpg)<br>
[Bike-Sharing-Demand-Kaggle-Competition-Dataset-Link](https://www.kaggle.com/competitions/bike-sharing-demand/data)

## Approach:

1. The dataset was downloaded and analyzed, with a focus on identifying its features and characteristics.
2. A model was trained using AutoGluon's Tabular Prediction on the given dataset and predictions were submitted to Kaggle for initial ranking. A baseline model was created with the default settings provided by AutoGluon's Tabular prediction.
3. An in-depth exploratory data analysis was conducted on the existing features, with the goal of determining how changes in the dataset would affect model performance. Feature engineering was incorporated to improve the performance of the model. This process was carried out using the default automated settings of AutoGluon's Tabular prediction.
4. Following the initial data analysis and feature engineering, multiple revised iterations of the model were carried out to further improve performance.
5. While performing the iterations mentioned in 4, the feature engineered dataset was used to train the AutoGluon's Tabular Prediciton model, with some of the hyperparameters being fine-tuned in order to further improve the model's performance.


## Getting Started (Resources)
* Clone this template repository `git clone git@github.com:udacity/nd009t-c1-intro-to-ml-project-starter.git` into AWS Sagemaker Studio (or local development).

<img src="img/sagemaker-studio-git1.png" alt="sagemaker-studio-git1.png" width="500"/>
<img src="img/sagemaker-studio-git2.png" alt="sagemaker-studio-git2.png" width="500"/>

* Visit the [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand) page. There you will see the overall details about the competition including overview, data, code, discussion, leaderboard, and rules. You will primarily be focused on the data and ranking sections.

### Dependencies

```
Python 3.7
pydantic 1.10.3
MXNet 1.8
Pandas >= 1.2.4
AutoGluon 0.2.0 
```

### Installation
For this project, it is highly recommended to use Sagemaker Studio from the course provided AWS workspace. This will simplify much of the installation needed to get started.

For local development, you will need to setup a jupyter lab instance.
* Follow the [jupyter install](https://jupyter.org/install.html) link for best practices to install and start a jupyter lab instance.
* If you have a python virtual environment already installed you can just `pip` install it.
```
pip install jupyterlab
```
* There are also docker containers containing jupyter lab from [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html).


## Information related to project specific files in this repository:<br><br>
**1. `project_notebook-bike_sharing_demand.ipynb`:** Jupyter notebook with code<br><br>
**2. `project_notebook-bike_sharing_demand.ipynb`:** HTML export of the jupyter notebbook<br><br>
**3. `project_report.md`:** (Markdown or PDF file of the report) A report was generated post-submission that meticulously examines the iterations that yielded the most significant improvement in model performance, along with a thorough explanation of the reasons for their effectiveness.<br><br>

## License
[License](LICENSE.txt)<br><br>

Thank you for taking the time to visit this repository!
