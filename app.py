import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# โหลดโมเดล
model_pack = joblib.load('ppfd_production_model.joblib')
et_model = model_pack['et_model']
hgbr_model = model_pack['hgbr_model']

# 🟢 หน้าเว็บ
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        try:
            sensors = []
            for i in range(1, 13):
                val = request.form.get(f'f{i}')
                if val == "" or val is None:
                    raise ValueError(f"Missing value f{i}")
                sensors.append(float(val))
        
            if any(s < 0 for s in sensors):
                raise ValueError("Sensor must be >= 0")
        
            x_log = np.log(np.array(sensors) + 1)
        
            f7_fvis = x_log[6] * x_log[11]
            x_input = np.append(x_log, f7_fvis).reshape(1, -1)
        
            p1 = et_model.predict(x_input)[0]
            p2 = hgbr_model.predict(x_input)[0]
        
            res = np.exp(p1 * 0.6 + p2 * 0.4) - 1
        
            result = round(float(res), 2)
        
        except Exception as e:
            result = str(e)

    return render_template_string("""
    <h2>PPFD Predictor</h2>
    <form method="post">
        {% for i in range(1,13) %}
            F{{i}}: <input name="f{{i}}" type="number" step="any" required><br>
        {% endfor %}
        <br>
        <button type="submit">Predict</button>
    </form>

    {% if result is not none %}
        <h3>Result: {{ result }}</h3>
    {% endif %}
    """, result=result)
    


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['sensors']
        x_log = np.log(np.array(data) + 1)
        
        # Interaction Term
        f7_fvis = x_log[6] * x_log[11]
        x_input = np.append(x_log, f7_fvis).reshape(1, -1)
        
        # Prediction
        p1 = et_model.predict(x_input)[0]
        p2 = hgbr_model.predict(x_input)[0]
        
        res = np.exp(p1 * 0.6 + p2 * 0.4) - 1
        return jsonify({"ppfd": round(float(res), 2)})
    except:
        return "Error", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
