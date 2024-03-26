#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
df=pd.read_csv('digital-eye.csv')
df


# In[3]:


df.drop(['Name'],axis=1)


# In[4]:


df.isnull().sum()


# In[5]:


# Specify numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Fill null values with the mean for numeric columns only
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())


# In[6]:


df.isnull().sum()


# In[7]:


# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Calculate the IQR for each numeric column
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# Define outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers
outliers = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).sum()

# Print the sum of outliers
print("Sum of outliers:")
print(outliers)


# In[8]:


numeric_df = df.select_dtypes(include='number')

# Calculate the IQR for each numeric column
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# Define outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Clean outliers by replacing them with the lower or upper bound
cleaned_df = numeric_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# Replace the numeric columns in the original DataFrame with the cleaned values
for column in cleaned_df.columns:
    df[column] = cleaned_df[column]


# In[9]:


# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Calculate the IQR for each numeric column
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# Define outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers
outliers = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).sum()

# Print the sum of outliers
print("Sum of outliers:")
print(outliers)


# In[10]:


numeric_df


# In[11]:


dd=numeric_df


# In[12]:


dd


# In[13]:


dd.info()


# In[14]:


dd.describe()


# In[127]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate Analysis
# Histogram for numerical variables
numerical_vars = ['Age','Sex','wearables','Duration','onlineplatforms','Nature',
                  'screenillumination','workingyears','hoursspentdailycurricular',
                  'hoursspentdailynoncurricular','Severityofcomplaints','RVIS',
                  'Ocularsymptomsobservedlately','Symptomsobservingatleasthalfofthetimes',
                  'Complaintsfrequency','frequencyofdryeyes',
                  'Schimers1Lefteye','Schimers1righteye','Schimers2Lefteye','Schimers2righteye']
dd[numerical_vars].hist(bins=20, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Variables')
plt.show()



# In[128]:


# Bar chart for categorical variables
categorical_vars = ['Sex', 'wearables', 'Nature', 'screenillumination']
for col in categorical_vars:
    sns.countplot(x=col, data=dd)
    plt.title(f'Bar Chart of {col}')
    plt.show()



# In[18]:


# Bivariate Analysis
# Scatter plot for numerical vs. numerical variables
sns.pairplot(dd[numerical_vars])
plt.suptitle('Pair Plot of Numerical Variables')
plt.show()



# In[25]:


# Correlation analysis
correlation_matrix = dd[numerical_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.figure(figsize=(10,6))
plt.show()


# In[26]:


# Outlier Detection
# Boxplot for numerical variables
plt.figure(figsize=(15, 10))
sns.boxplot(data=dd[numerical_vars])
plt.title('Boxplot of Numerical Variables')
plt.xticks(rotation=45)
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
X = dd.drop(['Schimers1Lefteye', 'Schimers1righteye', 'Schimers2Lefteye', 'Schimers2righteye'], axis=1)
y = dd[['Schimers1Lefteye', 'Schimers1righteye', 'Schimers2Lefteye', 'Schimers2righteye']]



# In[16]:


# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# In[17]:


# Build and train the model
model = RandomForestRegressor(
    n_estimators=40,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42  
)
model.fit(X_train, y_train)



# In[18]:


# Evaluate the model
y_pred = model.predict(X_test)




# In[35]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Model Used : RandomForestRegressor")
print( mse)
print( r2)
# Calculate MAPE
# Calculate absolute percentage error for each prediction
abs_percentage_error = np.abs((y_test - y_pred) / y_test) * 100

# Calculate median absolute percentage error
median_ape = np.median(abs_percentage_error)
print("Median Absolute Percentage Error (MdAPE): {:.2f}%".format(median_ape))
accuracy=100-median_ape
print("Accuracy")
print(accuracy)


# In[134]:


from scipy.stats import ttest_ind

# Example:
# Independent t-test to compare 'Age' between 'Sex' categories
male_age = dd[dd['Sex'] == 1]['Age']
female_age = dd[dd['Sex'] == 2]['Age']
t_statistic, p_value = ttest_ind(male_age, female_age)
print(f'T-statistic: {t_statistic}, p-value: {p_value}')


# In[135]:


# Assuming your dataset is stored in a DataFrame called dd

# 1. Histograms for Age and Working Years
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(dd['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(dd['workingyears'], bins=20, kde=True)
plt.title('Distribution of Working Years')
plt.xlabel('Working Years')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()




# In[136]:


# 2. Bar Chart for Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=dd, x='Sex')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()





# In[138]:


# 3. Box Plot for Duration of Online Platforms Usage
plt.figure(figsize=(8, 5))
sns.boxplot(data=dd, y='Duration')
plt.title('Duration of Online Platforms Usage')
plt.ylabel('Duration (hours)')
plt.show()



# In[139]:


# 4. Scatter Plot for Age vs. Duration of Online Platforms Usage
plt.figure(figsize=(8, 6))
sns.scatterplot(data=dd, x='Age', y='Duration')
plt.title('Age vs. Duration of Online Platforms Usage')
plt.xlabel('Age')
plt.ylabel('Duration (hours)')
plt.show()



# In[140]:


# 5. Heatmap for Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(dd.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()



# In[141]:


# 6. Bar Chart for Frequency of Complaints
plt.figure(figsize=(10, 6))
sns.countplot(data=dd, x='freqquencyofcomplaints')
plt.title('Frequency of Complaints')
plt.xlabel('Complaints')
plt.ylabel('Count')
plt.show()



# In[142]:


# 7. Stacked Bar Chart for Nature of Screen Illumination by Gender
plt.figure(figsize=(8, 6))
sns.countplot(data=dd, x='screenillumination', hue='Sex')
plt.title('Nature of Screen Illumination by Gender')
plt.xlabel('Nature of Screen Illumination')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()



# In[143]:


# 8. Line Plot for Average Nighttime Usage per Day by Age
plt.figure(figsize=(10, 6))
sns.lineplot(data=dd, x='Age', y='Avgnighttimeusageperday')
plt.title('Average Nighttime Usage per Day by Age')
plt.xlabel('Age')
plt.ylabel('Average Nighttime Usage (hours)')
plt.show()


# In[ ]:


# 9. Pairplot for Multiple Variable Comparison
sns.pairplot(dd)
plt.title('Pairplot of Numerical Variables')
plt.show()


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np

# Assuming X contains your features and y contains your target variables

# Convert y_train and y_test to numpy arrays
y_train_array = y_train.to_numpy()
y_test_array = y_test.to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor for each target variable
rf_models = []
rf_scores = []
for i in range(y.shape[1]):
    rf = RandomForestRegressor( n_estimators=40,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42  )
    rf.fit(X_train, y_train.iloc[:, i])
    rf_models.append(rf)
    y_pred_rf = rf.predict(X_test)
    # Calculate MAPE
    abs_percentage_error = np.abs((y_test_array[:, i] - y_pred_rf) / y_test_array[:, i]) * 100
    median_ape = np.median(abs_percentage_error)
    accuracy = 100 - median_ape
    rf_scores.append({
        "MAE": mean_absolute_error(y_test_array[:, i], y_pred_rf),
        "MSE": mean_squared_error(y_test_array[:, i], y_pred_rf),
        "R^2": r2_score(y_test_array[:, i], y_pred_rf),
        "Median APE": median_ape,
        "Accuracy": accuracy
    })

# Initialize and train Linear Regression for each target variable
lr_models = []
lr_scores = []
for i in range(y.shape[1]):
    lr = LinearRegression()
    lr.fit(X_train, y_train.iloc[:, i])
    lr_models.append(lr)
    y_pred_lr = lr.predict(X_test)
    # Calculate MAPE
    abs_percentage_error = np.abs((y_test_array[:, i] - y_pred_lr) / y_test_array[:, i]) * 100
    median_ape = np.median(abs_percentage_error)
    accuracy = 100 - median_ape
    lr_scores.append({
        "MAE": mean_absolute_error(y_test_array[:, i], y_pred_lr),
        "MSE": mean_squared_error(y_test_array[:, i], y_pred_lr),
        "R^2": r2_score(y_test_array[:, i], y_pred_lr),
        "Median APE": median_ape,
        "Accuracy": accuracy
    })

# Initialize and train Ridge Regression for each target variable
ridge_models = []
ridge_scores = []
for i in range(y.shape[1]):
    ridge = Ridge()
    ridge.fit(X_train, y_train.iloc[:, i])
    ridge_models.append(ridge)
    y_pred_ridge = ridge.predict(X_test)
    # Calculate MAPE
    abs_percentage_error = np.abs((y_test_array[:, i] - y_pred_ridge) / y_test_array[:, i]) * 100
    median_ape = np.median(abs_percentage_error)
    accuracy = 100 - median_ape
    ridge_scores.append({
        "MAE": mean_absolute_error(y_test_array[:, i], y_pred_ridge),
        "MSE": mean_squared_error(y_test_array[:, i], y_pred_ridge),
        "R^2": r2_score(y_test_array[:, i], y_pred_ridge),
        "Median APE": median_ape,
        "Accuracy": accuracy
    })

# Initialize and train Lasso Regression for each target variable
lasso_models = []
lasso_scores = []
for i in range(y.shape[1]):
    lasso = Lasso()
    lasso.fit(X_train, y_train.iloc[:, i])
    lasso_models.append(lasso)
    y_pred_lasso = lasso.predict(X_test)
    # Calculate MAPE
    abs_percentage_error = np.abs((y_test_array[:, i] - y_pred_lasso) / y_test_array[:, i]) * 100
    median_ape = np.median(abs_percentage_error)
    accuracy = 100 - median_ape
    lasso_scores.append({
        "MAE": mean_absolute_error(y_test_array[:, i], y_pred_lasso),
        "MSE": mean_squared_error(y_test_array[:, i], y_pred_lasso),
        "R^2": r2_score(y_test_array[:, i], y_pred_lasso),
        "Median APE": median_ape,
        "Accuracy": accuracy
    })

# Reporting scores
print("Random Forest Regressor Scores:")
for i, score in enumerate(rf_scores):
    print(f"Target {i + 1}:")
    print(f"{'MAE':<15} {score['MAE']:.2f}")
    print(f"{'MSE':<15} {score['MSE']:.2f}")
    print(f"{'R^2':<15} {score['R^2']:.2f}")
    print(f"{'Median APE':<15} {score['Median APE']:.2f}")
    print(f"{'Accuracy':<15} {score['Accuracy']:.2f}%")
    print()

print("\nLinear Regression Scores:")
for i, score in enumerate(lr_scores):
    print(f"Target {i + 1}:")
    print(f"{'MAE':<15} {score['MAE']:.2f}")
    print(f"{'MSE':<15} {score['MSE']:.2f}")
    print(f"{'R^2':<15} {score['R^2']:.2f}")
    print(f"{'Median APE':<15} {score['Median APE']:.2f}")
    print(f"{'Accuracy':<15} {score['Accuracy']:.2f}%")
    print()

print("\nRidge Regression Scores:")
for i, score in enumerate(ridge_scores):
    print(f"Target {i + 1}:")
    print(f"{'MAE':<15} {score['MAE']:.2f}")
    print(f"{'MSE':<15} {score['MSE']:.2f}")
    print(f"{'R^2':<15} {score['R^2']:.2f}")
    print(f"{'Median APE':<15} {score['Median APE']:.2f}")
    print(f"{'Accuracy':<15} {score['Accuracy']:.2f}%")
    print()

print("\nLasso Regression Scores:")
for i, score in enumerate(lasso_scores):
    print(f"Target {i + 1}:")
    print(f"{'MAE':<15} {score['MAE']:.2f}")
    print(f"{'MSE':<15} {score['MSE']:.2f}")
    print(f"{'R^2':<15} {score['R^2']:.2f}")
    print(f"{'Median APE':<15} {score['Median APE']:.2f}")
    print(f"{'Accuracy':<15} {score['Accuracy']:.2f}%")
    print()


# In[ ]:




