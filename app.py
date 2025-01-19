import streamlit as st
import numpy as np
import joblib

# 1) Load the trained model
model = joblib.load("model.joblib")

st.title("Medical Group Classifier")

st.write("""
Enter values for **9 features**, then click **Predict** to see the predicted group.
""")

# 2) Ask user for each of the 9 features
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)
f5 = st.number_input("Feature 5", value=0.0)
f6 = st.number_input("Feature 6", value=0.0)
f7 = st.number_input("Feature 7", value=0.0)
f8 = st.number_input("Feature 8", value=0.0)
f9 = st.number_input("Feature 9", value=0.0)

# 3) When user clicks, do the prediction
if st.button("Predict"):
    # Convert all inputs to a numpy array shape = (1, 9)
    X_new = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])
    
    predicted_group = model.predict(X_new)[0]
    st.write(f"**Predicted group:** {predicted_group}")