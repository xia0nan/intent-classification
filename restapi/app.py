from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from model import get_intent_nlp, get_intent_nlp_clustering

app = Flask(__name__)
api = Api(app)

class intentAPI(Resource):
    def get(self):
        question = request.form['question']
        intents, inference_time = get_intent_nlp(question, None)
        intents = intents.to_json(orient='records')
        data = {
            'intents': intents,
            'inference_time': inference_time
        }
        return jsonify(data)

api.add_resource(intentAPI, '/')

if __name__ == '__main__':
    app.run(debug=True)


# curl http://localhost:5000/ -d "question=this is a test" -X GET
# from requests import put, get
# put('http://localhost:5000/todo1', data={'question': 'this is a test'}).json()