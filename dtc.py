from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import math
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def categorize_time(row):
    # LateMidnight  00:00 - 05:50
    # Morning       06:00 - 11:59
    # Afternoon     12:00 - 17:59
    # EarlyMidnight 18:00 - 23:59
    hour = row.hour
    if 0 <= hour < 6:
        return 'LateMidnight'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'EarlyMidnight'
     
def categorize_month(row):
    month = row.month
    if month == 1:
        return 'Jan'
    elif month == 2:
        return 'Feb'
    elif month == 3:
        return 'March'
    elif month == 4:
        return 'April'
    elif month == 5:
        return 'May'
    elif month == 6:
        return 'June'
    elif month == 7:
        return 'July'
    elif month == 8:
        return 'Aug'
    elif month == 9:
        return 'Sept'   
    elif month == 10:
        return 'Oct'
    elif month == 11:
        return 'Nov'
    elif month == 12:
        return 'Dec'

def categorize_days(row):
    day = row.dayofweek 
    if day == 'Saturday' or day == 'Sunday':
        return 1
    else:
        return 0

# Load the cleaned training dataset
train_data = pd.read_csv('train.csv')

# Drop unnecessary columns not in test data
train_data.drop('Descript', inplace=True, axis=1) 
train_data.drop('Resolution', inplace=True, axis=1) 

# Removes locations on North Pole (removes outliers)
#train_data = train_data[train_data['Y'] != 90.0]
mean_Y = train_data['Y'].mean()
train_data['Y'] = np.where(train_data['Y'] == 90.0, mean_Y, train_data['Y'])


# Add New 'TimeOfDay' column to better categorize data
train_data['Dates'] = pd.to_datetime(train_data['Dates'], format='%Y-%m-%d %H:%M:%S')
train_data['TimeOfDay'] = train_data['Dates'].apply(categorize_time)

# Add new 'Month' column
train_data['Month'] = train_data['Dates'].apply(categorize_month)

# Add the 'IsWeekend' column based on the 'DayOfWeek' column
train_data['IsWeekend'] = train_data['DayOfWeek'].isin(['Saturday', 'Sunday'])

# Drop 'Dates' column as it is unnecessary
train_data.drop('Dates', inplace=True, axis=1) 

# Encoding categorical columns ==================================
# List of categorical columns
categorical_columns = ['Category', 'DayOfWeek', 'PdDistrict', 'Address', 'TimeOfDay', 'Month', 'IsWeekend']

# Encode each categorical column
for colName in categorical_columns:
    le = LabelEncoder()
    train_data[colName] = le.fit_transform(train_data[colName])

# Columns to train on
feature_columns = ['DayOfWeek', 'TimeOfDay', 'Month', 'IsWeekend', 'PdDistrict', 'Address', 'X', 'Y']
X_train = train_data[feature_columns]  # Features for training
y_train = train_data.Category  # Target variable for training

# Creating Model
model = DecisionTreeClassifier(max_depth=8)
model = model.fit(X_train, y_train)

# Load the cleaned test dataset
test_data = pd.read_csv('test.csv')

# Removes locations on North Pole (removes outliers)
mean_Y = test_data['Y'].mean()
test_data['Y'] = np.where(test_data['Y'] == 90.0, mean_Y, test_data['Y'])

mean_X = train_data['X'].mean()
train_data['X'] = np.where(train_data['X'] == -120.5, mean_X, train_data['X'])


# Apply the same preprocessing steps to the test data as the training data
test_data['Dates'] = pd.to_datetime(test_data['Dates'], format='%Y-%m-%d %H:%M:%S')
test_data['TimeOfDay'] = test_data['Dates'].apply(categorize_time)
test_data['Month'] = test_data['Dates'].apply(categorize_month)
test_data['IsWeekend'] = test_data['DayOfWeek'].isin(['Saturday', 'Sunday'])
test_data.drop('Dates', inplace=True, axis=1) 

# Encode categorical columns using the same LabelEncoder instances from training
for colName in categorical_columns[1:]:  # Skip 'Category' column
    test_data[colName] = le.transform(test_data[colName])

# Select the same feature columns as used in training
X_test_data = test_data[feature_columns]

# Make predictions on the test data
y_pred_test = model.predict(X_test_data)

# Calculate predicted probabilities for each class
y_pred_proba_test = model.predict_proba(X_test_data)

prediction = pd.DataFrame(y_pred_proba_test, columns=model.classes_)

# 884185 // results.csv
# 884261 // test.csv

# Add 'Id' column to the prediction DataFrame
prediction.insert(0, 'Id', range(len(test_data)))

# Define the desired column names
column_names = ["ARSON", "ASSAULT", "BAD CHECKS", "BRIBERY", "BURGLARY", "DISORDERLY CONDUCT", 
                "DRIVING UNDER THE INFLUENCE", "DRUG/NARCOTIC", "DRUNKENNESS", "EMBEZZLEMENT", 
                "EXTORTION", "FAMILY OFFENSES", "FORGERY/COUNTERFEITING", "FRAUD", "GAMBLING", 
                "KIDNAPPING", "LARCENY/THEFT", "LIQUOR LAWS", "LOITERING", "MISSING PERSON", 
                "NON-CRIMINAL", "OTHER OFFENSES", "PORNOGRAPHY/OBSCENE MAT", "PROSTITUTION", 
                "RECOVERED VEHICLE", "ROBBERY", "RUNAWAY", "SECONDARY CODES", "SEX OFFENSES FORCIBLE", 
                "SEX OFFENSES NON FORCIBLE", "STOLEN PROPERTY", "SUICIDE", "SUSPICIOUS OCC", 
                "TREA", "TRESPASS", "VANDALISM", "VEHICLE THEFT", "WARRANTS", "WEAPON LAWS"]

# Rename the columns
prediction.columns = ['Id'] + column_names

# Save the DataFrame to results.csv
prediction.to_csv('results.csv', index=False)