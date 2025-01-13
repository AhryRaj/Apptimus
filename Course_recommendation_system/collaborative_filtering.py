import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle

def load_data():
    try:
        interaction_file = './data/user_item_matrix.csv'
        purchase_count_file = './data/purchase_count.csv'
        enrollments_file = './data/enrollments_cleaned.csv'
        courses_file = './data/courses_cleaned.csv'

        # Load datasets
        User_item_matrix = pd.read_csv(interaction_file)
        Purchase_count_df = pd.read_csv(purchase_count_file)
        Enrollments_df = pd.read_csv(enrollments_file)
        Courses_df = pd.read_csv(courses_file)

        return User_item_matrix, Purchase_count_df, Enrollments_df, Courses_df
    except FileNotFoundError as E:
        raise FileNotFoundError(f"File not found: {E.filename}. Ensure the file path is correct.")
    except pd.errors.EmptyDataError:
        raise ValueError("One of the input files is empty. Please check the data files.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the data. Check the file format and content.")
    except Exception as E:
        raise Exception(f"An error occurred while loading data: {E}")

def prepare_data_for_svd(enrollment_df):
    try:
        reader = Reader(rating_scale=(0, 1))
        datas = Dataset.load_from_df(enrollment_df[['student_id', 'course_id', 'is_purchased']], reader)
        print("Data prepared for SVD:", datas)  # Print the prepared data
        return datas
    except KeyError as E:
        raise KeyError(f"Missing column in enrollments_df: {E}. Ensure the DataFrame contains 'student_id', 'course_id', and 'is_purchased'.")
    except Exception as E:
        raise Exception(f"An error occurred while preparing data for SVD: {E}")

def train_svd_model(datas):
    try:
        train_set, test_set = train_test_split(datas, test_size=0.2, random_state=42)  # random_state can be any integer
        svd_algorithm = SVD()
        svd_algorithm.fit(train_set)
        return svd_algorithm, train_set, test_set
    except ValueError as E:
        raise ValueError(f"Error splitting data: {E}. Ensure the data is properly formatted.")
    except Exception as E:
        raise Exception(f"An error occurred while training the SVD model: {E}")

def evaluate_model(svd_algorithm, test_set):
    try:
        predictions = svd_algorithm.test(test_set)

        y_true = [int(predict.r_ui) for predict in predictions]
        y_prediction = [int(predict.est >= 0.5) for predict in predictions]  # user-course interaction >= 0.5

        Precision = precision_score(y_true, y_prediction)
        Recall = recall_score(y_true, y_prediction)
        F1 = f1_score(y_true, y_prediction)

        Accuracy = (np.sum(np.array(y_true) == np.array(y_prediction)) / len(y_true))

        print(f'Precision: {Precision:.2f}')
        print(f'Recall: {Recall:.2f}')
        print(f'F1-Score: {F1:.2f}')
        print(f'Accuracy: {Accuracy:.2f}')

        return Precision, Recall, F1, Accuracy
    except ValueError as E:
        raise ValueError(f"Error in evaluation: {E}")
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero occurred during evaluation. Check the test data.")
    except Exception as E:
        raise Exception(f"An error occurred while evaluating the model: {E}")

def recommend_courses(student_ids, useritem_matrix, svd_algorithm, course_df, purchases_count_df, top_n=5):
    try:
        if student_ids not in useritem_matrix.index:
            raise ValueError(f"Student ID {student_ids} not found in user-item matrix.")

        purchased_courses = useritem_matrix.loc[student_ids][useritem_matrix.loc[student_ids] > 0].index.tolist()

        valid_courses = purchases_count_df[purchases_count_df['purchase_count'] >= 10].index.tolist()

        course_scores = {}
        for course_id in valid_courses:
            if course_id not in purchased_courses:
                predicted_score = svd_algorithm.predict(student_ids, course_id).est
                course_scores[course_id] = predicted_score

        if not course_scores:
            raise ValueError(f"No courses available for recommendation for student {student_ids}.")

        sorted_courses = sorted(course_scores, key=course_scores.get, reverse=True)
        recommend_course = course_df[course_df['course_id'].isin(sorted_courses[:top_n])]['title'].tolist()

        return recommend_course
    except KeyError as E:
        raise KeyError(f"Key error while processing recommendations: {E}. Ensure data consistency across DataFrames.")
    except ValueError as E:
        raise ValueError(f"Value error during recommendation: {E}")
    except Exception as E:
        raise Exception(f"An error occurred while recommending courses: {E}")

def save_model(svd_algorithm, model_filename="collaborative_filtering_model.pkl"):
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(svd_algorithm, f)
        print(f"\nModel saved as {model_filename}")
    except IOError as E:
        raise IOError(f"IO error while saving the model: {E}")
    except Exception as E:
        raise Exception(f"An error occurred while saving the model: {E}")

def load_model(model_filename="collaborative_filtering_model.pkl"):
    try:
        with open(model_filename, 'rb') as f:
            svd_algorithm = pickle.load(f)
        print(f"\nModel loaded from {model_filename}")
        return svd_algorithm
    except FileNotFoundError:
        print(f"Model file {model_filename} not found. Recomputing...")
        return None
    except pickle.UnpicklingError:
        raise ValueError("Error unpickling the model. The file might be corrupted.")
    except Exception as E:
        raise Exception(f"An error occurred while loading the model: {E}")

# Main execution
if __name__ == "__main__":
    try:
        user_item_matrix, purchase_count_df, enrollments_df, courses_df = load_data()

        data = prepare_data_for_svd(enrollments_df)

        svd, trainset, testset = train_svd_model(data)

        precision, recall, f1, accuracy = evaluate_model(svd, testset)

        save_model(svd, "./models/collaborative_filtering_model.pkl")

        student_id = 892
        title = 'IELTS'

        print(f"\nTesting recommendations for student {student_id}...")
        recommended_courses = recommend_courses(student_id, user_item_matrix, svd, courses_df,
                                                purchase_count_df, top_n=5)

        print(f"\nRecommended courses for student {student_id}: {recommended_courses}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
