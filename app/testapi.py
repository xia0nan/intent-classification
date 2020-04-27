import requests

url = 'http://54.169.47.108/get_intent_nlp_clustering'
question = "open account"
data = {"question":question}

response = requests.post(url, json = data)

response.json()