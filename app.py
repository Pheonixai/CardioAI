from flask import Flask, render_template, request, make_response
import joblib
import numpy as np
from xhtml2pdf import pisa
from io import BytesIO
from datetime import datetime
import time
import os

app = Flask(__name__)

# Load your trained model
model = joblib.load('cardio_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()  # Start timer

        # Collect features
        features = [
            float(request.form['oldpeak']),
            float(request.form['thalach']),
            float(request.form['ca']),
            float(request.form['cp']),
            float(request.form['thal']),
            float(request.form['age']),
            float(request.form['trestbps']),
            float(request.form['exang'])
        ]
        final_input = np.array([features])

        # Predict
        prediction = model.predict(final_input)[0]
        probabilities = model.predict_proba(final_input)[0]
        confidence = round(max(probabilities) * 100, 2)
        class0 = round(probabilities[0] * 100, 2)
        class1 = round(probabilities[1] * 100, 2)
        result_text = "⚠️ Risk of Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease Detected"

        # Advice based on prediction
        if prediction == 1:
            advice = "Please consult a cardiologist as soon as possible. Consider lifestyle changes and regular monitoring."
        else:
            advice = "Keep up the healthy lifestyle! Regular exercise, good diet, and checkups are recommended."

        # Processing time
        duration = round(time.time() - start_time, 2)

        # Current date & time
        current_time = datetime.now().strftime("%A, %d %B %Y at %I:%M %p")

        return render_template(
            "result.html",
            prediction_text=result_text,
            confidence=confidence,
            class0=class0,
            class1=class1,
            confidence_note="Confidence scores are based on model probability.",
            user_inputs=request.form,
            processing_time=duration,
            current_time=current_time,
            advice=advice
        )

    except Exception as e:
        return f"❌ Error during prediction: {e}"

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        user_data = request.form.to_dict()

        # Re-extract features
        features = [
            float(user_data['oldpeak']),
            float(user_data['thalach']),
            float(user_data['ca']),
            float(user_data['cp']),
            float(user_data['thal']),
            float(user_data['age']),
            float(user_data['trestbps']),
            float(user_data['exang'])
        ]
        final_input = np.array([features])
        prediction = model.predict(final_input)[0]
        probabilities = model.predict_proba(final_input)[0]
        confidence = round(max(probabilities) * 100, 2)
        class0 = round(probabilities[0] * 100, 2)
        class1 = round(probabilities[1] * 100, 2)
        result_text = "⚠️ Risk of Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease Detected"

        # Advice again
        advice = "Please consult a cardiologist URGENTLY." if prediction == 1 else "Maintain your healthy habits."

        # Current time
        current_time = datetime.now().strftime("%A, %d %B %Y at %I:%M %p")

        # Render template
        rendered = render_template(
            'report_template.html',
            name="CardioAI Medical Report",
            result_text=result_text,
            confidence=confidence,
            class0=class0,
            class1=class1,
            data=user_data,
            current_time=current_time,
            advice=advice
        )

        pdf_buffer = BytesIO()
        pisa_status = pisa.CreatePDF(rendered, dest=pdf_buffer)
        pdf_buffer.seek(0)

        if pisa_status.err:
            return "❌ Error generating PDF"

        response = make_response(pdf_buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=CardioAI_Report.pdf'
        return response

    except Exception as e:
        return f"❌ Error during report generation: {e}"

if __name__ == '__main__':
    app.run(

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=True)

        #debug=False
        )
