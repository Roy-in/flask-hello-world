import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "PPFD API Running"
    
# โหลดโมเดล
model_pack = joblib.load('ppfd_production_model.joblib')
et_model = model_pack['et_model']
hgbr_model = model_pack['hgbr_model']

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
        return str(round(float(res), 2))
    except:
        return "Error", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
