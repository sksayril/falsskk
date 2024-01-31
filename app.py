from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pandas as pd
import difflib

app = Flask(__name__)
api = Api(app)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
data = pd.read_pickle('kitpot.searchingdata14.pkl')
titles = data['DATA'].tolist()

lowercase_titles = [str(title).lower() if isinstance(title, str) else '' for title in titles]
data = list(filter(None, lowercase_titles))

def find_closest_match(input_word, data):
    if input_word in data:
        return [input_word]

    closest_match = difflib.get_close_matches(input_word, data)
    return closest_match

def process_search_term(search_term, data):
    words = search_term.split(' ')
    processed_words = []

    for word in words:
        
        if word.lower() in data:
            processed_words.append(word)
        else:
            closest_match = find_closest_match(word.lower(), data)
            processed_words.append(closest_match[0] if closest_match else word)

    processed_sentence = ' '.join(processed_words)
    return processed_sentence

@app.route('/get_closest_match', methods=['GET'])
def get_closest_match():
    search_term = request.args.get('search_key', default='', type=str)
    
    if not search_term:
        response = {
            'message': 'Search term is empty.',
        }
        return jsonify(response)

    processed_sentence = process_search_term(search_term, data)
    
    if processed_sentence:
        response = {
            'result': processed_sentence,
        }
    else:
        response = {
            'message': 'No similar words found.',
        }

    return jsonify(response)

try:
    with open('converted_data.json', 'r', encoding='utf-8') as file:
        products = json.load(file)

    for product in products:
        product['model'] = product['model'].lower()
        product['title'] = product['title'].lower()

    for product in products:
        product['feedback'] = []

except FileNotFoundError:
    raise Exception("Error: Products data file ('converted_data.json') not found.")

embedding_file = 'product_embeddings.npy'
try:
    encoded_descriptions = np.load(embedding_file)
except FileNotFoundError:
    descriptions = [f"{product['model']} {product['title']}" for product in products]
    encoded_descriptions = model.encode(descriptions)
    np.save(embedding_file, encoded_descriptions)

try:
    with open('interaction_state.json', 'r') as file:
        interaction_state = json.load(file)
except FileNotFoundError:
    interaction_state = {}



def update_interaction_state(interaction_state):
    with open('interaction_state.json', 'w') as file:
        json.dump(interaction_state, file)


def get_recommendations(query, num_recommendations=5, excluded_indices=[]):
    query = query.lower() 
    query_embedding = model.encode([query])
    model_query = query.split(' ')[0]

    
    model_match_indices = [i for i, product in enumerate(products) if product['model'] == model_query]

    if model_match_indices:
        model_match_indices = np.array(model_match_indices)
        similarities = cosine_similarity(query_embedding, encoded_descriptions[model_match_indices]).flatten()
        sorted_indices = model_match_indices[np.argsort(similarities)[::-1]]
    else:
        
        similarities = cosine_similarity(query_embedding, encoded_descriptions).flatten()
        sorted_indices = np.argsort(similarities)[::-1]

    recommended_products = []
    for index in sorted_indices:
        if len(recommended_products) >= num_recommendations:
            break
        if index not in excluded_indices:
            recommended_products.append(products[index])

    return recommended_products



class Recommendation(Resource):
    def get(self):
        query = request.args.get('query')
        if not query:
            return jsonify({"error": "Query parameter 'query' is required"}), 400

        excluded_indices = interaction_state.get(query, [])
        recommended_products = get_recommendations(query, num_recommendations=5, excluded_indices=excluded_indices)

        if not recommended_products:
            return jsonify({"message": f"No recommendations found for the model: {query.split(' ')[0]}"})
        else:
            result = [{"_id": product['_id'],"title": product['title'], "model": product['model']} for product in recommended_products]
            return jsonify(result)


class Feedback(Resource):
    def post(self):
        data = request.get_json()

        query = data.get('query')
        feedback_dict = data.get('feedback')

        if not query or not feedback_dict:
            return jsonify({"error": "Both 'query' and 'feedback' parameters are required"}), 400

        excluded_indices = interaction_state.get(query, [])

        for i, feedback in feedback_dict.items():
            recommended_product = get_recommendations(query, num_recommendations=5, excluded_indices=excluded_indices)[int(i) - 1]
            if feedback is not None:
                recommended_product['feedback'].append(feedback)
                excluded_indices.append(products.index(recommended_product))

        interaction_state[query] = excluded_indices
        update_interaction_state(interaction_state)

        return jsonify({"message": "Feedback updated. Thank you!"})



api.add_resource(Recommendation, '/recommendation')
api.add_resource(Feedback, '/feedback')

if __name__ == '__main__':
    app.run(debug=True)
