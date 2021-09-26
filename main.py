from flask import Flask
from flask import request, jsonify
from model.model import Classifier
import torch
import traceback
import json
from utils.utils import predict

device = torch.device('cpu')
app = Flask(__name__)
model = Classifier().to(device)
model.load_state_dict(torch.load('data/best_model', map_location=torch.device('cpu')))


@app.route('/classify', methods=['POST'])
def classify() -> None:
    if request.method == 'POST':
        try:
            input_data = json.loads(request.data)
            text = input_data['text']
            result = predict(model, text)
            return jsonify(result)
        except:
            return jsonify({"trace": traceback.format_exc()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

