import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import pickle
import cohere
import os
import pypdf


from flask import Blueprint, Flask
import os


views = Blueprint('views', __name__)

from flask import Flask, request, jsonify, render_template
COHEREKEY = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key=COHEREKEY)



directory = './context/'
all_documents = []

for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        filepath = os.path.join(directory, filename)
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        all_documents.extend(documents)

cohere_docs = []
for i, doc in enumerate(all_documents):
    cohere_docs.append({
        "id": f"doc_{i}",
        "data": {"text": doc.page_content}
    })


# Add the user query

# Generate the response


# Display the response
@views.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("query")
    response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": user_input}],
    documents=cohere_docs,
     )
    result = response.message.content[0].text
    return jsonify(result)



def create_app():
    app = Flask('website')
    app.config['SECRET_KEY'] = "BVCJEFHHKBWEIBVKEDBVUBVKJWDVUBEWJVHEN KERBGFBGKBWDIHVKRWDNVIERHFBNERKHV"

    app.register_blueprint(views, url_prefix='/')


    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
