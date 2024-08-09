import sqlite3
import pandas as pd

# Define the path to your CSV files
#PJ4 - change this
csv_student_path = r'C:\Users\Sarah Son Kim\class24\NU-VIRT-DATA-PT-02-2024-U-LOLC\02-Homework\Project4_Team5\Resources\Student_performance_data.csv'

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('mydatabase-pj4.sqlite', timeout=10)
c = conn.cursor()

# Create the price_cut table
c.execute('''
    CREATE TABLE IF NOT EXISTS StudentPerformance (
        StudentID INTEGER NOT NULL PRIMARY KEY,
        Age INTEGER NOT NULL,
        Gender INTEGER NOT NULL,
        Ethnicity INTEGER NOT NULL,
        ParentalEducation INTEGER NOT NULL,
        StudyTimeWeekly FLOAT NOT NULL,
        Absences INTEGER NOT NULL,
        Tutoring INTEGER NOT NULL,
        ParentalSupport INTEGER NOT NULL,
        Extracurricular INTEGER NOT NULL,
        Sports INTEGER NOT NULL,
        Music INTEGER NOT NULL,
        Volunteering INTEGER NOT NULL,
        GPA FLOAT NOT NULL,
        GradeClass INTEGER NOT NULL
        )
''')

# Commit the changes
conn.commit()

# Read the price_cut CSV file into a pandas DataFrame
df_student = pd.read_csv(csv_student_path)
df_student['GradeClass'] = df_student['GradeClass'].astype(int)

df_student.to_sql('StudentPerformance', conn, if_exists='replace', index=False)

# Commit the changes
conn.commit()

c.execute("SELECT * FROM StudentPerformance LIMIT 5")
rows = c.fetchall()
print(pd.DataFrame(rows, columns=[description[0] for description in c.description]))
# Close the connection
conn.close()
