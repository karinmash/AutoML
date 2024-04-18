# AutoML Web Application
a web application for processing classification problems using Streamlit and Pycaret. 🔎🔎

Upload your dataset.
View a summary of your dataset.
Visualize correlation matrix and values distribution.
Clean your data.
Apply data preprocessing techniques.
Train regression/classification models.
Evaluate your trained models.
Download your models.

# How to use it?
* Drag and drop your csv file
* Assign your target variable
* 
* Choose your machine learning model
* Let the app do all the job for you :)

# Overview
* This is a supervised Machine Learning based Project in which you can Upload a Dataset of your choice,
* View the Charts and Bar graphs related to the Dataset,
* Click on the TRAIN button,
* Finally get the Scores, Code and Report for your Model :)


# Using the app:
Select a dataset
Select an outcome variable and supervised ML task (regression or classification) to perform
Press the "Run AutoML" button to perform AutoML and generate Python code for the best ML pipeline
Note: The running time for pipeline optimization and evaluation time per iteration is limited to to 10 minutes max. In practice, AutoML with TPOT should be run with multiple instances in parallel for much longer (hours or days). You can modify this limit by modifying the "Maximum running time" Streamlit slider in app.py.






# Steps involved in the Project
### Upload the Dataset
  * Drag and drop your csv file.
  * Assign your target variable.
    ![image](https://github.com/karinmash/AutoML/assets/111049027/3b6ac8eb-ce69-4373-9289-364878596549)


### Exploratory Data Analysis
  * Select Visualizing the Data.
  * Assign your target variable.

First I installed all the necessary libraries required for this Project.

Then I imported the Data by reading csv file using read.csv() Method.

Then I dropped the Invoice ID Column because we don't need it in analysis.

After that I listed down all the columns in the Dataset by df.columns Method.

Then I used df.shape Method to look for the rows and columns in the Data.

Then I look for the Info of the Dataset by using df.info() Method.

Cleaning the Data

First I start by describing the Data by using df.describe() Method.

Then I converted Date Column to Pandas Date and Time DataType.

And After that I extracted Year, Month, Day from the Date.

Then I listed down all the unique values of categorical columns.

And Finally I verified the null values in the Dataset by using df.isna().sum()

Visualizing the Data

Subplots of Distribution of Unit Price, Ratings and Gross Income
 
