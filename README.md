# AutoML Web Application 🔎🔎
An End-to-End Machine Learning Web Application for processing Classification problems which, for a dataset from the user, performs data cleaning, data preprocessing, classification and optimization of hyperparameters for classification algorithms and displays the best algorithm and its values.

![recording.gif](https://github.com/karinmash/AutoML/blob/e073a7df31849cff38352435cf771710524ad110/recording.gif)


recording.gif

https://github.com/karinmash/AutoML/blob/3734364debdc4a14225af3c9b3e4a9d28ca241c7/recording.gif


Currently supported for csv and excel files. The application relies on these two excellent libraries for machine learning:
 * Streamlit: https://github.com/streamlit/streamlit
 * Pycaret: https://github.com/pycaret/pycaret
                            
---

## Features
### 1. Upload the Dataset
  * Drag and drop your csv file.
  * Assign your target variable.

### 2. Exploratory Data Analysis options:
  * Show shape.
  * Show data type.
  * Show missing values
  * Description
  * Show columns
  * Show selected columns
  * Show Correlation Heatmap
  * Show Value Counts
  * Show Unique Values
  * Show ydata profiling
    
### 3. Data Cleaning options:
* Remove duplicates
* Drop specific columns
* Handle missing data
  * Drop Missing Values (Drop all rows with Nan values, Drop all columns with Nan values)
  * Missing Values Imputation (Mean Imputation, Median Imputation, Most frequent Imputation, Random Imputation)

### 4. Data Preprocessing options:
* Scale your dataset (MinMaxScaler)
* Encode your dataset (Label Encoder, One Hot Encoder, Ordinal Encoder)
* Balance your dataset (Over Sampling, Under Sampling, Combined)

### 5. Model Training preforms:
* Training and Comparing all available Machine Learning Algorithm automatically.
* Hyperparameter tuning for 5 best models and Comparing them.
* Returning the best model.
* Best Model Result Visualization (plots)
* Save whole Machine Learning Pipeline of the best model as pickle file.
    


    







   





 
