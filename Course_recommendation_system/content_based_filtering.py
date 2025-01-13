import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_course_data():
    try:
        course_file = './data/courses_cleaned.csv'
        courses = pd.read_csv(course_file)
        return courses
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}. Ensure the file path is correct.")
    except pd.errors.EmptyDataError:
        raise ValueError("The courses data file is empty. Please check the file.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the courses data. Check the file format and content.")
    except Exception as e:
        raise Exception(f"An error occurred while loading course data: {e}")

def compute_course_similarity(courses):
    try:
        # Combine 'title' and 'short_description' for more detailed similarity
        courses['combined_text'] = (
            courses['title'].str.lower().fillna('') + ' ' +
            courses['short_description'].fillna('') + ' ' +
            courses['course_category'].str.lower().fillna('')
        )

        # Use TF-IDF Vectorizer to convert text to vector space and compute cosine similarity
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(courses['combined_text'].fillna(''))

        # Compute cosine similarity
        course_similarity = cosine_similarity(tfidf_matrix)

        # Convert the similarity matrix to a DataFrame for easier access
        return pd.DataFrame(course_similarity, index=courses['course_id'].astype(str), columns=courses['course_id'].astype(str))
    except KeyError as e:
        raise KeyError(f"Missing required column in courses data: {e}")
    except ValueError as e:
        raise ValueError(f"Error computing TF-IDF: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while computing course similarity: {e}")

def save_model(course_similarity_df, model_filename="content_based_filtering_model.pkl"):
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(course_similarity_df, f)
        print(f"\nCourse similarity model saved as {model_filename}")
    except Exception as e:
        raise Exception(f"An error occurred while saving the model: {e}")

def load_model2(model_filename="content_based_filtering_model.pkl"):
    try:
        with open(model_filename, 'rb') as f:
            course_similarity_df = pickle.load(f)
        print(f"\nCourse similarity model loaded from {model_filename}")
        return course_similarity_df
    except FileNotFoundError:
        print(f"Model file {model_filename} not found. Recomputing...")
        return None
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {e}")

def get_content_recommendations(course_id, course_similarity_df, top_n=5):
    try:
        course_id = str(course_id)  # Ensure the course_id is a string

        if course_id not in course_similarity_df.columns:
            raise ValueError(f"Course ID {course_id} not found in course similarity DataFrame.")

        # Get the top N most similar courses based on cosine similarity
        similar_courses = course_similarity_df[course_id].sort_values(ascending=False).index[1:top_n + 1]

        return list(similar_courses)
    except KeyError as e:
        raise KeyError(f"Key error while accessing similarity data: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while getting content recommendations: {e}")

def test_content_based_recommendation(titles, top_n=5):
    try:
        model_filename = "./models/content_based_filtering_model.pkl"
        course_similarity_df = load_model2(model_filename)

        if course_similarity_df is None:
            courses = load_course_data()
            course_similarity_df = compute_course_similarity(courses)
            save_model(course_similarity_df, model_filename)
        else:
            courses = load_course_data()

        # Find course_id for the given title
        course_row = courses[courses['title'].str.lower() == titles.lower()]
        if course_row.empty:
            raise ValueError(f"Course title '{titles}' not found in courses data.")

        course_id = course_row['course_id'].values[0]

        # Get content-based recommendations
        recommendations = get_content_recommendations(course_id=course_id, course_similarity_df=course_similarity_df,
                                                      top_n=top_n)

        if not recommendations:
            print("No recommendations available.")
            return

        # Display the recommended courses along with their titles
        print(f"\nRecommended Courses for '{titles}':")
        for recommended_course_id in recommendations:
            title_row = courses.loc[courses['course_id'] == int(recommended_course_id), 'title']
            if not title_row.empty:
                print(f"Course ID: {recommended_course_id}, Title: {title_row.values[0]}")
            else:
                print(f"Course ID: {recommended_course_id}, Title: Not found in course data")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An error occurred during content-based recommendation: {e}")

# Main execution
if __name__ == "__main__":
    student_id = 892  # Example student_id for testing
    title = 'IELTS'  # Example course title
    test_content_based_recommendation(title, top_n=5)
