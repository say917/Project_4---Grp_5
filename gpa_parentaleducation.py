import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Import and read the Student_performance_data.csv
student_df = pd.read_csv('C:\\Users\\Sarah Son Kim\\class24\\NU-VIRT-DATA-PT-02-2024-U-LOLC\\02-Homework\\Project4_Team5\\Resources\\Student_performance_data.csv')

# Drop the non-beneficial ID columns, 'StudentID', 'Age', 'Gender', 'Ethnicity'
student_df = student_df.drop(columns=['StudentID', 'Age', 'Gender', 'Ethnicity'])

# Focus on only the 'ParentalEducation' and 'GPA' columns
parental_education = student_df['ParentalEducation']
gpa = student_df['GPA']

# Ensure GPA is within the range 2.0 to 4.0
gpa = gpa.clip(lower=2.0, upper=4.0)

# Calculate the Pearson correlation coefficient between GPA and ParentalEducation
correlation, p_value = pearsonr(parental_education, gpa)
print(f"Correlation between GPA and ParentalEducation: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")

# Visualize the correlation with a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(parental_education, gpa, color='blue')
plt.xlabel('Parental Education Level')
plt.ylabel('GPA')
plt.title('Correlation between GPA and ParentalEducation')
plt.grid(True)
plt.show()
