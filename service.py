from flask import Flask, request, jsonify
import psycopg2
import json
import os
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

documents = None
text_chunks = None
embeddings = None
vector_store = None
llm_model = None
chain = None
documents_directory = 'TUG'
llm_model_path = './LLM_model/Mistral/mistral-7b-instruct-v0.1.Q4_K_M'
conn = psycopg2.connect(
    dbname='forge',
    user='ffbot',
    password='ffpass',
    host='localhost',
    port='5432'
)

def load_documents(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def create_embeddings():
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # embeddings = SentenceTransformer(model_name)
    # return embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
    return embeddings

def create_vector_store(text_chunks, embeddings):
    # vectors = embeddings.encode(text_chunks)
    # dim = vectors.shape[1]
    # index = faiss.IndexFlatL2(dim)
    # index.add(vectors)
    # return index
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

def create_llms_model(model_path):
    # tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    llm = CTransformers(model='LLM_model/Microsoft/Phi-3-mini-4k-instruct-fp16.gguf', config={'max_new_tokens':8192, 'temperature':0.01})
    return llm

def start_model():
    documents = load_documents(documents_directory)
    text_chunks = split_text_into_chunks(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(text_chunks, embeddings)
    llm_model = create_llms_model(llm_model_path)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm_model, chain_type='stuff', retriever=vector_store.as_retriever(search_kwargs={"k": 2}), memory=memory)

def conversation_chat(user_email, smdg_terminal_code, query):
    cursor = conn.cursor()
    cursor.execute("SELECT chat_history FROM user_chats WHERE user_email = %s AND smdg_terminal_code = %s",
                   (user_email, smdg_terminal_code))
    result = cursor.fetchone()
    
    if result is not None:
        chat_history = result[0]
    else:
        chat_history = []

    # result = chain({"question": query, "chat_history": chat_history})
    query_result = { "answer": "Answer for " + query }

    update_chat_history(result, cursor, chat_history, user_email, smdg_terminal_code, query, query_result["answer"])
    conn.commit()
    cursor.close()
    return query_result["answer"]

def update_chat_history(result, cursor, chat_history, user_email, smdg_terminal_code, query, answer):
    chat_history.append({'question': query, 'answer': answer})
    if len(chat_history) > 30:
        chat_history = chat_history[-30:]
    
    if result is not None:
        print("record_updating")
        cursor.execute("UPDATE user_chats SET chat_history = %s WHERE user_email = %s AND smdg_terminal_code = %s",
                       (json.dumps(chat_history), user_email, smdg_terminal_code))
    else:
        print("record_inserting")
        cursor.execute("INSERT INTO user_chats (user_email, smdg_terminal_code, chat_history) VALUES (%s, %s, %s)",
                       (user_email, smdg_terminal_code, json.dumps(chat_history)))

def get_chat_history(user, smdg):
    cursor = conn.cursor()
    cursor.execute("SELECT chat_history FROM user_chats WHERE user_email = %s AND smdg_terminal_code = %s",
                   (user, smdg))
    result = cursor.fetchone()
    
    if result is not None:
        chat_history = result[0]
    else:
        chat_history = []

    conn.commit()
    cursor.close()
    return chat_history

# update_chat_history('user@example.com', 'T123', 'What is a proforma?', 'A proforma is...')


app = Flask(__name__)

@app.route("/query/<user>/<smdg>", methods=["GET"])
def query(user, smdg):
    query = request.args.get("query")
    return conversation_chat(user, smdg, query)

@app.route("/history/<user>/<smdg>", methods=["GET"])
def history(user, smdg):
    return jsonify(get_chat_history(user, smdg)), 200

if __name__ == "__main__":
    app.run(debug=True)