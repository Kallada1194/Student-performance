import gradio as gr
import joblib
import pandas as pd

# Load the model and unique values
model = joblib.load('model.joblib')
unique_values = joblib.load('unique_values.joblib')

# Define the prediction function
def predict(student_id, age, gender, ethnicity, parental_education, study_time_weekly, absences, tutoring, parental_support, extracurricular, sports, music, volunteering, gpa):
    # Convert inputs to appropriate types
    student_id = int(student_id)
    age = int(age)
    gender = int(gender)
    ethnicity = int(ethnicity)
    parental_education = int(parental_education)
    study_time_weekly = float(study_time_weekly)
    absences = int(absences)
    tutoring = int(tutoring)
    parental_support = int(parental_support)
    extracurricular = int(extracurricular)
    sports = int(sports)
    music = int(music)
    volunteering = int(volunteering)
    gpa = float(gpa)
    
    # Prepare the input array for prediction
    input_data = pd.DataFrame({
        'StudentID': [student_id],
        'Age': [age],
        'Gender': [gender],
        'Ethnicity': [ethnicity],
        'ParentalEducation': [parental_education],
        'StudyTimeWeekly': [study_time_weekly],
        'Absences': [absences],
        'Tutoring': [tutoring],
        'ParentalSupport': [parental_support],
        'Extracurricular': [extracurricular],
        'Sports': [sports],
        'Music': [music],
        'Volunteering': [volunteering],
        'GPA': [gpa]
    })
    
    # Perform the prediction
    prediction = model.predict(input_data)
    
    return prediction[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Student ID  "),
        gr.Textbox(label="Age : The age of the students ranges from 15 to 18 years."),
        gr.Textbox(label="Gender : 0 represents Male and 1 represents Female"),
        gr.Textbox(label="Ethnicity : 0: Caucasian , 1: African American , 2: Asian , 3: Other"),
        gr.Textbox(label="Parental Education : 0: None , 1: High School , 2: Some College , 3: Bachelor's , 4: Higher"),
        gr.Textbox(label="Study Time Weekly : ranging from 0 to 20"),
        gr.Textbox(label="Absences : ranging from 0 to 30"),
        gr.Textbox(label="Tutoring : 0 indicates No and 1 indicates Yes"),
        gr.Textbox(label="Parental Support : 0: None , 1: Low , 2: Moderate , 3: High , 4: Very High"),
        gr.Textbox(label="Extracurricular : 0 indicates No and 1 indicates Yes"),
        gr.Textbox(label="Sports : 0 indicates No and 1 indicates Yes"),
        gr.Textbox(label="Music : 0 indicates No and 1 indicates Yes"),
        gr.Textbox(label="Volunteering : 0 indicates No and 1 indicates Yes"),
        gr.Textbox(label="GPA : Grade Point Average on a scale from 2.0 to 4.0")
    ],
    outputs="text",
    title="Student Performance Predictor",
    description="Enter student details to predict their grade class."
)

# Launch the app
interface.launch()
