from flask import Flask, render_template, request, jsonify
import pandas as pd
from hybrid_filtering import hybrid_recommendation, load_data, load_models

app = Flask(__name__)

# Load the collaborative and content-based models
def load_collaborative_and_content_based_models():
    try:
        Collaborative_model, Content_based_model = load_models()
        if Collaborative_model is None or Content_based_model is None:
            raise ValueError("Both collaborative and content-based models must be properly loaded.")
        return Collaborative_model, Content_based_model
    except FileNotFoundError as E:
        raise FileNotFoundError(f"Model file not found: {E.filename}. Please check model paths.")
    except Exception as E:
        raise Exception(f"Error while loading models: {E}")

# Load data
try:
    user_item_matrix, purchase_count_df, courses_df = load_data()
    enrollments_df = pd.read_csv('./data/enrollments_cleaned.csv')
except FileNotFoundError as e:
    raise FileNotFoundError(f"File not found: {e.filename}. Ensure the file paths are correct.")
except pd.errors.EmptyDataError:
    raise ValueError("One or more data files are empty. Please verify the data files.")
except Exception as e:
    raise Exception(f"An error occurred while loading data: {e}")

# Load the collaborative and content-based models
try:
    collaborative_model, content_based_model = load_collaborative_and_content_based_models()
except Exception as e:
    raise Exception(f"Error initializing the app: {e}")

@app.route('/')
def home():
    """
    Render the home page with inputs for student ID and course title.
    """
    try:
        student_ids = enrollments_df['student_id'].unique().tolist()
        course_titles = courses_df['title'].unique().tolist()
        course_categories = courses_df['course_category'].dropna().unique().tolist()
        return render_template('index.html', student_ids=student_ids, course_titles=course_titles, course_categories=course_categories)
    except KeyError as E:
        return f"Key Error: {E}. Check if the required columns exist in the data.", 500
    except Exception as E:
        return f"An error occurred while loading the home page: {E}", 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """
    Handle the recommendation request and return results as a table.
    """
    try:
        student_id = request.form.get('student_id')
        course_title = request.form.get('course_title')
        category_filter = request.form.get('category_filter')

        # Validate student ID
        try:
            student_id = int(student_id)
        except ValueError:
            return jsonify({'error': "Invalid Student ID. Please enter a valid number or select from the dropdown."}), 400

        if student_id not in user_item_matrix.index:
            return jsonify({'error': f"Student ID {student_id} not found in the dataset."}), 400

        if course_title not in courses_df['title'].values:
            return jsonify({'error': f"Course title '{course_title}' not found in the dataset."}), 400

        # Generate recommendations
        recommended_courses, hybrid_scores = hybrid_recommendation(
            student_id=student_id,
            title=course_title,
            collaborative_model=collaborative_model,
            content_based_model=content_based_model,
            useritem_matrix=user_item_matrix,
            courses_df=courses_df,
            purchase_count_df=purchase_count_df,
            top_n=10,
            alpha=0.5,  # Weight for collaborative filtering
            category_filter=category_filter if category_filter else None
        )

        if not recommended_courses:
            return jsonify({'error': "No recommendations could be generated. Please check the inputs or data."}), 400

        # Prepare data for the response
        results = []
        for course in recommended_courses:
            try:
                course_title = courses_df.loc[courses_df['course_id'] == int(course), 'title'].values[0]
                course_category = courses_df.loc[courses_df['course_id'] == int(course), 'course_category'].values[0]
                results.append({'course_id': course, 'course_title': course_title, 'course_category': course_category})
            except KeyError:
                results.append({'course_id': course, 'course_title': "Unknown", 'course_category': "Unknown"})

        return jsonify({'results': results})
    except ValueError as ve:
        return jsonify({'error': f"Value Error: {ve}"}), 400
    except FileNotFoundError as fe:
        return jsonify({'error': f"File Error: {fe}"}), 500
    except Exception as E:
        return jsonify({'error': f"An unexpected error occurred: {E}"}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Critical error: {e}")
