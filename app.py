import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_loan(age, gender, income, experience, loan_amount,
                 interest_rate, credit_history, credit_score, previous_loan):

    gender_num = 0 if gender == "Male" else 1
    previous_loan_num = 1 if previous_loan == "Yes" else 0
    loan_percent = loan_amount / income if income > 0 else 0

    new_data = pd.DataFrame(
        [[age, gender_num, income, experience, loan_amount,
          interest_rate, loan_percent, credit_history,
          credit_score, previous_loan_num]],
        columns=scaler.feature_names_in_
    )

    new_scaled = scaler.transform(new_data)
    pred = model.predict(new_scaled)[0]
    prob = model.predict_proba(new_scaled)[0][1]

    if pred == 1:
        return f"Loan Approved ✅ (confidence: {prob*100:.1f}%)"
    return f"Loan Not Approved ❌ (confidence: {(1-prob)*100:.1f}%)"

app = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Age", value=30),
        gr.Radio(["Male", "Female"], label="Gender", value="Male"),
        gr.Number(label="Person Income", value=60000),
        gr.Number(label="Employee Experience (years)", value=5),
        gr.Number(label="Loan Amount", value=10000),
        gr.Number(label="Loan Interest Rate", value=10),
        gr.Number(label="Credit History (length in years)", value=5),
        gr.Number(label="Credit Score", value=700),
        gr.Radio(["Yes", "No"], label="Previous loan default on file?", value="No"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🏦 Loan Approval Prediction",
    description="Enter the applicant's details to predict loan approval. Trained on a synthetic practice dataset."
)

app.launch()