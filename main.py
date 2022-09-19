from flask import Flask, jsonify, request
from enum import Enum
from tools_server.controllers.controllers import run_model_controller
from Model_IA import Model

app = Flask(__name__)

class Paths(Enum):
    MODEL='/model'
    RUN_MODEL = '/model/run'

@app.route('/', methods=['GET'])
def index():
    return jsonify({'Message': 'Welcome to API REST tutorial'})

@app.route('/model/run', methods=['POST'])
def runModel():
    bl, msg = run_model_controller(request)
    if not bl:
        return jsonify({'Error': msg})

    currency, start_day, num_days = request.json['currency'], request.json['start_day'], request.json['num_days']

    model = Model(currency, start_day, num_days)

    model.predict_asset()

    return jsonify({'Message': 'Predict successfully'})

@app.route('/model/retraining', methods=['POST'])
def retrainingModel():
    bl, msg = run_model_controller(request)
    if not bl:
        return jsonify({'Error': msg})

    currency, start_day, num_days = request.json['currency'], request.json['start_day'], request.json['num_days']

    model = Model(currency, start_day, num_days)

    model.retraining_model()

    return jsonify({'Message': 'Predict successfully'})

if __name__ == '__main__':
    app.run(debug=True)