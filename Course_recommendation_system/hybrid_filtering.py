import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_models():
    """
    Load the saved collaborative and content-based models.
    """
    try:
        with open("./models/collaborative_filtering_model.pkl", 'rb') as f:
            collaborative_model = pickle.load(f)
        print("\nCollaborative filtering model loaded successfully.")
    except FileNotFoundError:
        print("Collaborative filtering model not found.")
        collaborative_model = None
    except Exception as E:
        raise Exception(f"An error occurred while loading the collaborative model: {E}")

    try:
        with open("./models/content_based_filtering_model.pkl", 'rb') as f:
            content_based_model = pickle.load(f)
        print("\nContent-based filtering model loaded successfully.")
    except FileNotFoundError:
        print("Content-based filtering model not found.")
        content_based_model = None
    except Exception as E:
        raise Exception(f"An error occurred while loading the content-based model: {E}")

    return collaborative_model, content_based_model

def load_data():
    """
    Load necessary datasets for recommendation.
    """
    try:
        useritem_matrix = pd.read_csv('./data/user_item_matrix.csv')
        purchase_count_df = pd.read_csv('./data/purchase_count.csv')
        courses_df = pd.read_csv('./data/courses_cleaned.csv')
        return useritem_matrix, purchase_count_df, courses_df
    except FileNotFoundError as E:
        raise FileNotFoundError(f"File not found: {E.filename}. Ensure the file path is correct.")
    except pd.errors.EmptyDataError:
        raise ValueError("One or more data files are empty. Please check the files.")
    except Exception as E:
        raise Exception(f"An error occurred while loading data: {E}")

def hybrid_recommendation(student_id, title, collaborative_model, content_based_model,
                          useritem_matrix, courses_df, purchase_count_df, top_n=5, alpha=0.6, category_filter=None):
    """
    Generate hybrid recommendations by combining collaborative and content-based scores.
    The recommendations can be filtered by category.
    """
    try:
        # Validate inputs
        if student_id not in useritem_matrix.index:
            raise ValueError(f"Student ID {student_id} not found in user-item matrix.")

        if collaborative_model is None or content_based_model is None:
            raise ValueError("Both collaborative and content-based models must be loaded.")

        # Collaborative recommendations
        purchased_courses = useritem_matrix.loc[student_id][useritem_matrix.loc[student_id] > 0].index.tolist()
        valid_courses = purchase_count_df[purchase_count_df['purchase_count'] >= 10].index.tolist()

        collaborative_scores = {}
        for course_id in valid_courses:
            if course_id not in purchased_courses:
                try:
                    predicted_score = collaborative_model.predict(student_id, course_id).est
                    collaborative_scores[course_id] = predicted_score
                except Exception as E:
                    print(f"Error predicting score for Course ID {course_id}: {E}")

        # Content-based recommendations
        course_row = courses_df[courses_df['title'].str.lower() == title.lower()]
        if course_row.empty:
            raise ValueError(f"Course title '{title}' not found in courses data.")

        input_course_id = str(course_row['course_id'].values[0])
        if input_course_id not in content_based_model.columns:
            raise ValueError(f"Course ID {input_course_id} not found in content similarity model.")

        content_scores = content_based_model[input_course_id].to_dict()

        # Combine scores using weighted sum
        Hybrid_scores = {}
        for course in set(list(collaborative_scores.keys()) + list(content_scores.keys())):
            collab_score = collaborative_scores.get(course, 0)
            content_score = content_scores.get(course, 0)
            Hybrid_scores[course] = alpha * collab_score + (1 - alpha) * content_score

        # Sort courses by hybrid scores
        sorted_courses = sorted(Hybrid_scores, key=Hybrid_scores.get, reverse=True)

        # Filter out purchased courses, the input course, and limit to top_n
        Recommended_courses = [
            course for course in sorted_courses
            if course not in purchased_courses and course != input_course_id
        ][:top_n]

        # Apply category filter if provided
        if category_filter:
            Recommended_courses = [
                course for course in Recommended_courses
                if courses_df.loc[courses_df['course_id'] == int(course), 'course_category'].values[0] == category_filter
            ]

        # Print recommendations
        print(f"\nHybrid Recommendations for user {student_id} and course '{title}':")
        for course in Recommended_courses:
            try:
                course_title = courses_df.loc[courses_df['course_id'] == int(course), 'title'].values[0]
                print(f"Course ID: {course}, Title: {course_title}, Score: {Hybrid_scores[course]:.2f}")
            except KeyError:
                print(f"Course ID: {course}, Title: Not found in course data.")

        # Calculate and print correlation between input course and each recommended course
        input_course_vector = content_based_model.loc[input_course_id].values.reshape(1, -1)
        print(f"\nCorrelation between input course '{title}' and each recommended course:")
        for course in Recommended_courses:
            try:
                recommended_vector = content_based_model.loc[str(course)].values.reshape(1, -1)
                correlation = cosine_similarity(input_course_vector, recommended_vector).item()
                recommended_title = courses_df.loc[courses_df['course_id'] == int(course), 'title'].values[0]
                print(f" - Course ID: {course}, Title: {recommended_title}, Correlation: {correlation:.2f}")
            except KeyError as E:
                print(f" - Error: Could not calculate correlation for Course ID {course}. Reason: {E}")

        return Recommended_courses, Hybrid_scores
    except ValueError as E:
        print(f"Value Error: {E}")
    except Exception as E:
        print(f"An error occurred during hybrid recommendation: {E}")
        return [], {}

def save_hybrid_model(hybrid_model, model_filename="hybrid_filtering_model.pkl"):
    """
    Save the hybrid recommendation model as a pickle file.
    """
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(hybrid_model, f)
        print(f"\nHybrid model saved as {model_filename}")
    except Exception as E:
        raise Exception(f"An error occurred while saving the hybrid model: {E}")

if __name__ == "__main__":
    try:
        # Load saved models
        collaborative_model, content_based_model = load_models()

        # Load data
        user_item_matrix, Purchase_count_df, courses_df = load_data()

        # Generate hybrid recommendations
        test_student_id = 892
        test_title = 'IELTS'
        top_n_recommendations = 10
        alpha_weight = 0.5  # Weight for collaborative filtering

        print("\nGenerating hybrid recommendations...")
        recommended_courses, hybrid_scores = hybrid_recommendation(
            student_id=test_student_id,
            title=test_title,
            collaborative_model=collaborative_model,
            content_based_model=content_based_model,
            useritem_matrix=user_item_matrix,
            courses_df=courses_df,
            purchase_count_df=Purchase_count_df,
            top_n=top_n_recommendations,
            alpha=alpha_weight
        )

        # Save the hybrid model
        save_hybrid_model(hybrid_scores, "./models/hybrid_filtering_model.pkl")
    except Exception as e:
        print(f"An error occurred in the main program: {e}")