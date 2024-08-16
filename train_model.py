import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

# Import and read the Student_performance_data.csv
student_df = pd.read_csv('C:\\Users\\Sarah Son Kim\\class24\\NU-VIRT-DATA-PT-02-2024-U-LOLC\\02-Homework\\Project4_Team5\\Resources\\Student_performance_data.csv')

# Drop the non-beneficial ID columns
student_df = student_df.drop(columns=['StudentID', 'Age', 'Gender', 'Ethnicity'])

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

# Check for multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = features
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print("Variance Inflation Factor (VIF):")
print(vif_data)

# If needed, you can drop features with high VIF before proceeding
# Example: If 'Volunteering' has a high VIF, you may consider dropping it

# Re-scale after dropping columns if necessary
# X_scaled = scaler.fit_transform(X)

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Model: Ridge Regression
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

# Visualize the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients_ridge.index, data=coefficients_ridge)
plt.title('Ridge Regression Coefficients')
plt.grid(True)
plt.show()

# Save the model
with open('model_ridge.pkl', 'wb') as model_file:
    pickle.dump(model_ridge, model_file)
