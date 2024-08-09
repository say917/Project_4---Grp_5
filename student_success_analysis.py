import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your data
csv_student_path = r'C:\Users\Sarah Son Kim\class24\NU-VIRT-DATA-PT-02-2024-U-LOLC\02-Homework\Project4_Team5\Resources\Student_performance_data.csv'
data = pd.read_csv(csv_student_path)

# Define features (independent variables) and target (dependent variable)
# Predict GPA based on other features
X = data[['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
          'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GradeClass']]
y = data['GPA']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the features (optional, but recommended for better performance)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Display the model coefficients (weights assigned to each feature)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Visualization: Example of StudyTimeWeekly vs. GPA
plt.scatter(data['StudyTimeWeekly'], data['GPA'])
plt.xlabel('Study Time Weekly')
plt.ylabel('GPA')
plt.title('Study Time vs GPA')
plt.show()
