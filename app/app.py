from flask import Flask, jsonify, render_template, request
import pandas as pd
import logging
from infer import get_intent_nlp, get_intent_nlp_clustering

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("test.html")

@app.route('/get_intent_nlp', methods=['POST'])
def ajax_api():
    question = request.json['question']
    intents, inference_time = get_intent_nlp(question, None)
    intents = intents.to_json(orient='records')
    data = {
        'intents': intents,
        'inference_time': inference_time
    }
    return jsonify(data)

# %% main
if __name__ == '__main__':
    app.run(threaded=False, debug=True, host='127.0.0.1', port=5002)

# %% gunicorn
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
