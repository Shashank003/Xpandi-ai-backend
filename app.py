from flask import Flask, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI

load_dotenv(override=True)
app = Flask("__name__")
CORS(app)


@app.route("/", methods=['POST'])
def index():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}),400
    data = request.get_json()
    return jsonify({"name": data["firstname"]+" "+data["lastname"]})


client = OpenAI()
def generate_response(query):
    stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system","content": "Use the following User Input (or previous conversaton if needed) to answer the users question."},{"role": "user", "content": query}],
    stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""

@app.route("/stream",methods=["POST"])
def stream_response():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}),400
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Request must contain query"}),400
    
    return Response(stream_with_context(generate_response(data["query"])), content_type='text/plain')
    


if __name__ == '__main__':
    app.run(debug=False)