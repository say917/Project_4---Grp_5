import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
csv_student_path = r'C:\Users\Sarah Son Kim\class24\NU-VIRT-DATA-PT-02-2024-U-LOLC\02-Homework\Project4_Team5\Resources\Student_performance_data.csv'
data = pd.read_csv(csv_student_path)

# Define features (independent variables) and target (dependent variable)
# Predict GradeClass based on other features
X = data[['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
          'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']]
y = data['GradeClass']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the features (optional, but recommended for better performance)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Initialize and train the model (using Logistic Regression for classification)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Display the classification report
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['A', 'B', 'C', 'D', 'F']))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'B', 'C', 'D', 'F'], yticklabels=['A', 'B', 'C', 'D', 'F'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot StudyTimeWeekly vs. GradeClass to visualize relationship
plt.figure(figsize=(10, 6))
sns.boxplot(x='GradeClass', y='StudyTimeWeekly', data=data, palette='Set2')
plt.xlabel('Grade Class')
plt.ylabel('Study Time Weekly (hours)')
plt.title('Study Time Weekly by Grade Class')
plt.show()
