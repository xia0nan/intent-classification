from flask import Flask, jsonify, render_template, request
import pandas as pd
import logging
from infer import get_intent_nlp_clustering

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("test.html")

@app.route('/get_intent_nlp_clustering', methods=['POST'])
def ajax_api():
    question = request.json['question']
    intents = get_intent_nlp_clustering(question)
    data = {
        'intents': intents
    }
    return jsonify(data)

# %% main
if __name__ == '__main__':
    app.run(threaded=False, debug=True, host='0.0.0.0', port=5002, use_reloader=False)

# %% gunicorn
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
