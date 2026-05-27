#----------------------------------------------------------------------------------
# Logistic Regression Dataset
# @author: David Ortega Lozano
# @date: 2026-05-12
# @version: 0.1
# @description: This module loads the breast cancer dataset, splits it into 
# training and testing sets, and scales the features using standardization. The 
# resulting datasets are used in the industrialized case experiment defined in 
# objectives/industrialized_case.py.
#----------------------------------------------------------------------------------

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset.
data = load_breast_cancer()

x = data.data
y = data.target

# Split the dataset into training and testing sets with stratification to maintain
# class balance.
global x_train, x_test, y_train, y_test 
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale the features using standardization to improve the performance of the 
# logistic regression model.
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)