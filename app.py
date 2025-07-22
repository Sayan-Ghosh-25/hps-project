import numpy as np
import pickle
from flask import Flask, render_template, request

# ─── Load Trained Model ───
model = pickle.load(open('Model.pkl', 'rb'))

# ─── Home Page ───
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('Health.html', result=None)

# ─── Predict Route ───
@app.route('/predict', methods=['POST'])
def prediction():
    try:
        # Extract Form Data
        AGE = int(request.form.get('AGE'))
        SEX = int(request.form.get('SEX'))
        AN = int(request.form.get('AN'))
        CP = int(request.form.get('CP'))
        DB = int(request.form.get('DB'))
        EF = int(request.form.get('EF'))
        HBP = int(request.form.get('HBP'))
        PLT = int(request.form.get('PLT'))
        SC = float(request.form.get('SC'))
        SS = int(request.form.get('SS'))
        SMK = int(request.form.get('SMK'))
        T = int(request.form.get('T'))

        # Prepare Data and Predict
        input_data = np.array([AGE, AN, CP, DB, EF, HBP, PLT, SC, SS, SEX, SMK, T]).reshape(1, -1)
        print(input_data)
        prediction = model.predict_proba(input_data)[0][1]
        print(prediction)

        if prediction >= 0.7:
            result = f"⚠️ High Risk! (Death Chance: {100 * prediction:.0f}%)"
        elif prediction >= 0.3:
            result = f"💊 Medium Risk! (Death Chance: {100 * prediction:.0f}%)"
        else:
            result = f"✅ Low Risk! (Death Chance: {100 * prediction:.0f}%)"

        return render_template('Health.html', result=result)

    except Exception as e:
        return render_template('Health.html', result="❌ Invalid Input!: " + str(e))

# ─── Run Server ───
if __name__ == '__main__':
    app.run(debug=True, port=8080)