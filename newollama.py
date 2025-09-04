
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


import cohere



from flask import Blueprint
import os


views = Blueprint('views', __name__)

from flask import Flask, request, jsonify, render_template
COHEREKEY = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key='EEYfgu69cPTLtyj8riog2s8HLY5ClqyyWde6EHrS')


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

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(all_documents)

# Embeddings + vector database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_store")


@views.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("query")
    results = db.similarity_search(user_input, k=5)
    context = "\n\n".join([r.page_content for r in results])
    response = co.chat(
    model="command-r",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ],
    max_tokens=1000
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)



