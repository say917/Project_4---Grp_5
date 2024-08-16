import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# Import and read the Student_performance_data.csv
student_df = pd.read_csv('C:\\Users\\Sarah Son Kim\\class24\\NU-VIRT-DATA-PT-02-2024-U-LOLC\\02-Homework\\Project4_Team5\\Resources\\Student_performance_data.csv')

# Drop the non-beneficial ID columns
student_df = student_df.drop(columns=['StudentID', 'Age', 'Gender', 'Ethnicity'])

# Check for outliers in Absences
#plt.figure(figsize=(10, 6))
#sns.boxplot(x=student_df['Absences'])
#plt.title('Boxplot of Absences')
#plt.show()

# Visualize the distribution of Absences
#plt.figure(figsize=(10, 6))
#sns.histplot(student_df['Absences'], bins=20, kde=True)
#plt.title('Distribution of Absences')
#plt.show()

# Correlation plot between Absences and GPA
#plt.figure(figsize=(10, 6))
#sns.scatterplot(x=student_df['Absences'], y=student_df['GPA'])
#plt.xlabel('Absences')
#plt.ylabel('GPA')
#plt.title('Scatter Plot of GPA vs Absences')
#plt.grid(True)
#plt.show()

# Non-linear transformation (e.g., log) of Absences
student_df['Log_Absences'] = np.log1p(student_df['Absences'])

# Features and Target Variable
features = ['ParentalEducation', 'StudyTimeWeekly', 'Tutoring', 'ParentalSupport', 
            'Extracurricular', 'Sports', 'Music', 'Volunteering', 'Log_Absences']

X = student_df[features]
y = student_df['GPA']

# Ensure GPA is within the range 2.0 to 4.0
y = y.clip(lower=2.0, upper=4.0)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Model 1: Ridge Regression
model_ridge = Ridge(alpha=10.0)
model_ridge.fit(X_train, y_train)

# Predict GPA using the test set with Ridge
y_pred_ridge = model_ridge.predict(X_test)
y_pred_ridge = np.clip(y_pred_ridge, 2.0, 4.0)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")
print(f"Ridge Regression - R-squared: {r2_ridge}")

coefficients_ridge = pd.DataFrame(model_ridge.coef_, features, columns=['Coefficient'])
print("Ridge Regression Coefficients:")
print(coefficients_ridge)

# Model 2: Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=1)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
y_pred_rf = np.clip(y_pred_rf, 2.0, 4.0)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
#print(f"Random Forest - Mean Squared Error: {mse_rf}")
#print(f"Random Forest - R-squared: {r2_rf}")

importances_rf = pd.DataFrame({'Feature': features, 'Importance': model_rf.feature_importances_})
importances_rf = importances_rf.sort_values(by='Importance', ascending=False)

#plt.figure(figsize=(10, 6))
#sns.barplot(x='Importance', y='Feature', data=importances_rf)
#plt.title('Feature Importances - Random Forest')
#plt.show()

# Save models
with open('model_ridge.pkl', 'wb') as model_file:
    pickle.dump(model_ridge, model_file)

with open('model_rf.pkl', 'wb') as model_file:
    pickle.dump(model_rf, model_file)
