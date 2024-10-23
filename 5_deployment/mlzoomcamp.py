from flask import Flask
from flask import request
import pickle

app = Flask("predict")

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"



model_file = f"model2.bin"
dv_file = f"dv.bin"

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]

    return str(y_pred)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)