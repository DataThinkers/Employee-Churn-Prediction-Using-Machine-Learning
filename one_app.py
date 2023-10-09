# Organize the layout
import streamlit as st
import pandas as pd
import pickle

st.sidebar.title("Select One")
app_selection = st.sidebar.selectbox("Select App", ["Single Prediction", "Prediction Using Test File"])

if app_selection == "Single Prediction":
    # Load the pre-trained model
    with open('data.pkl','rb') as f:
            data = pickle.load(f)
            
    with open('pipeline.pkl','rb') as f:
        pipeline = pickle.load(f)
    
    # Function to show prediction result
    def show_prediction():
        p1 = float(e1)
        p2 = float(e2)
        p3 = float(e3)
        p4 = float(e4)
        p5 = float(e5)
        p6 = float(e6)
        p7 = float(e7)
        p8 = str(e8)
        p9 = str(e9)
    
        sample = pd.DataFrame({
            'satisfaction_level': [p1],
            'last_evaluation': [p2],
            'number_project': [p3],
            'average_montly_hours': [p4],
            'time_spend_company': [p5],
            'Work_accident': [p6],
            'promotion_last_5years': [p7],
            'departments': [p8],
            'salary': [p9]
        })
    
        result = pipeline.predict(sample)
        
        if result == 1:
            st.write("An employee may leave the organization.")
        else:
            st.write("An employee may stay with the organization.")
    
    # Streamlit app
    st.title("Predicting Employee Churn Using Machine Learning")
    
    # Employee data input fields
    
    
    e1 = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
    e2 = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
    e3 = st.slider("Number of projects assigned to", 1, 10, 5)
    e4 = st.slider("Average monthly hours worked", 50, 300, 150)
    e5 = st.slider("Time spent at the company", 1, 10, 3)
    e6 = st.radio("Whether they have had a work accident", [0, 1])
    e7 = st.radio("Whether they have had a promotion in the last 5 years", [0, 1])
    
    options = ('sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
           'RandD', 'accounting', 'hr', 'management')
    e8 = st.selectbox("Department name", options)
    
    options1 = ('low','meduim','high')
    e9 = st.selectbox("Salary category", options1)
    
    # Predict button
    if st.button("Predict"):
        show_prediction()

else:
    # Content for App 2
    # Function to process data
    def process_data(data):
        # Load the model and perform predictions
        with open('pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        
        result = pipeline.predict(data)
        
        # Assign predictions based on result
        y_pred = ["An employee may leave the organization." if pred == 1 
                  else "An employee may stay with the organization." for pred in result]
        
        # Add predicted target to the data
        data['Predicted_target'] = y_pred
        return data
    
    # Streamlit app
    st.title("Predicting Employee Churn Using Machine Learning")
    
    # Button to upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load data from CSV
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.replace('\n', '')
            data.rename(columns={'Departments ': 'departments'}, inplace=True)
            data = data.drop_duplicates()
            
            # Process the data
            processed_data = process_data(data)
            
            # Save the processed data to a CSV file
            st.write("Processed Data:")
            st.write(processed_data)
            st.write("Saving the processed data...")
            processed_data.to_csv('processed_data.csv', index=False)
            st.success("Data saved successfully!")
        except Exception as e:
            st.error(f"Failed to open file: {e}")
    
