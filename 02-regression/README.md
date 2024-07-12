Reference from [DataTalksClub](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/02-regression)


## 2.1 Car price prediction project
This project is about the creation of a model for helping users to predict car prices. The dataset was obtained from [this 
kaggle competition](https://www.kaggle.com/CooperUnion/cardataset).

**Project plan:**

* Prepare data and Exploratory data analysis (EDA)
* Use linear regression for predicting price
* Understanding the internals of linear regression 
* Evaluating the model with RMSE
* Feature engineering  
* Regularization 
* Using the model 


## 2.2 Data preparation

**Pandas attributes and methods:** 

* pd.read_csv(<file_path_string>) - read csv files 
* df.head() - take a look of the dataframe 
* df.columns - retrieve colum names of a dataframe 
* df.columns.str.lower() - lowercase all the letters 
* df.columns.str.replace(' ', '_') - replace the space separator 
* df.dtypes - retrieve data types of all features 
* df.index - retrieve indices of a dataframe


## 2.3 Exploratory data analysis
**Pandas attributes and methods:** 

* df[col].unique() - returns a list of unique values in the series 
* df[col].nunique() - returns the number of unique values in the series 
* df.isnull().sum() - returns the number of null values in the dataframe 

**Matplotlib and seaborn methods:**

* %matplotlib inline - assure that plots are displayed in jupyter notebook's cells
* sns.histplot() - show the histogram of a series 
   
**Numpy methods:**
* np.log1p() - applies log transformation to a variable and adds one to each result.

Long-tail distributions usually confuse the ML models, so the recommendation is to transform the target variable distribution to a normal one whenever possible. 


## 2.4 Setting up the validation framework
In general, the dataset is split into three parts: training, validation, and test. For each partition, we need to obtain feature matrices (X) and y vectors of targets. First, the size of partitions is calculated, records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices. 

**Pandas attributes and methods:** 

* `df.iloc[]` - returns subsets of records of a dataframe, being selected by numerical indices
* `df.reset_index()` - restate the orginal indices 
* `del df[col]` - eliminates target variable 

**Numpy methods:**

* `np.arange()` - returns an array of numbers 
* `np.random.shuffle()` - returns a shuffled array
* `np.random.seed()` - set a seed 
