from flask import Flask, jsonify, request
import json

app = Flask(__name__)

from genetic_algorithm_pipeline import create_initial_population, crossover, mutate, insert, delete, limit_phrases, evaluate_fitness

@app.route('/get_data', methods=['GET'])
def get_data():
    # Logic to retrieve and return the data
    data = {"epochs": 3, "params": "Your data here"}
    return jsonify(data)

@app.route('/post_data', methods=['POST'])
def post_data():
    # Logic to handle the posted data
    content = request.json
    # Process the content...
    return jsonify({"updated": True})

if __name__ == '__main__':
    app.run(debug=True)