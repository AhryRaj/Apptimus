import pandas as pd
import numpy as np


# Load datasets
courses = pd.read_csv('./data/courses.csv')
enrollments = pd.read_csv('./data/enrollments.csv')
course_categories = pd.read_csv('./data/course_categories.csv')

# Display the first few rows of each dataset to understand the structure and contents
courses_head = courses.head()
enrollments_head = enrollments.head()
course_categories_head = course_categories.head()

print(courses_head, enrollments_head, course_categories_head)

# selecting some columns to form new dataframe
courses_cleaned = courses[['id','title', 'short_description','course_category_id']]
enrollments_cleaned = enrollments[['student_id','course_id','paid_amount']]
course_categories_cleaned = course_categories[['id','title']]

#renaming the column
courses_cleaned.rename(columns={'id': 'course_id'}, inplace=True)
course_categories_cleaned.rename(columns={'id': 'course_category_id', 'title': 'course_category'}, inplace=True)


# Handle missing values for other important columns
courses_cleaned = courses_cleaned.dropna(subset=['course_id', 'title','short_description','course_category_id']).fillna('Unknown')
enrollments_cleaned = enrollments_cleaned.dropna(subset=['course_id', 'student_id'])
course_categories_cleaned = course_categories_cleaned.dropna(subset=['course_category_id','course_category'])

# Replace null values in 'paid_amount' with 0
enrollments_cleaned['paid_amount'] = enrollments_cleaned['paid_amount'].fillna(0)

#creating a new column
enrollments_cleaned['is_purchased'] = enrollments_cleaned['paid_amount'].apply(lambda x: 0 if x == 0 else 1)

#changing data type
enrollments_cleaned['student_id'] = enrollments_cleaned['student_id'].astype(int)

#merging two datasets
courses_cleaned = pd.merge(courses_cleaned, course_categories_cleaned, on='course_category_id', how='left')

#change all the letters in the column to lower case
courses_cleaned['short_description'] = courses_cleaned['short_description'].str.lower()

# Add a new column 'curriculum' with random values
curriculum_levels = ["beginner", "intermediate", "advanced"]
courses_cleaned["curriculum"] = np.random.choice(curriculum_levels, size=len(courses_cleaned))

#check whether there are any null values
print(enrollments_cleaned.isna().sum())
print(courses_cleaned.isna().sum())
print(course_categories_cleaned.isna().sum())

# Verify the cleaned data
print(courses_cleaned.head(), enrollments_cleaned.head(), course_categories_cleaned.head())

# Save the cleaned files to new CSV files
courses_cleaned.to_csv('./data/courses_cleaned.csv', index=False)
enrollments_cleaned.to_csv('./data/enrollments_cleaned.csv', index=False)

#create user_item_matrix
def create_user_item_matrix(enrollment_file):
    # Load enrollment data
    enrollment = pd.read_csv(enrollment_file)

    # Create user-item interaction matrix
    useritem_matrix = enrollment.pivot_table(index='student_id',
                                             columns='course_id',
                                             values='is_purchased',
                                             fill_value=0)
    return useritem_matrix

# Generate and save user-item matrix
user_item_matrix = create_user_item_matrix('./data/enrollments_cleaned.csv')
user_item_matrix.to_csv('./data/user_item_matrix.csv', index=True)
