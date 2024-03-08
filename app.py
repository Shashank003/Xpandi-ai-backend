from flask import Flask, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain_community.llms import OpenAI as LangchainOpenAI

load_dotenv(override=True)
app = Flask("__name__")
CORS(app)

model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
prompt = hub.pull("hwchase17/openai-tools-agent")
client = OpenAI()
llm = LangchainOpenAI(streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()], temperature=0)

@tool
def create_image(description: str) -> str:
    """Creating images based on the description of the image"""
    response = client.images.generate(
        model="dall-e-3",
        prompt=description,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    # print(response)
    image_url = response.data[0].url
    # print(image_url)
    return image_url

tools = [create_image]

agent = create_openai_tools_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

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
    
@app.route("/answerQuestion",methods=["POST"])
def answer_question():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}),400
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Request must contain query"}),400
    
    answer = agent_executor.invoke(
    {
        "input": data["query"]
    }
    )
    print(answer)
    return Response(answer["output"])

if __name__ == '__main__':
    app.run(debug=True)