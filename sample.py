import os
from transformers import LongformerForMaskedLM
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    model = LongformerForMaskedLM.from_pretrained(model_path)
    return model

def main():
    # Load documents
    documents_directory = 'TUG'
    documents = load_documents(documents_directory)

    # Split documents into chunks
    text_chunks = split_text_into_chunks(documents)

    # Create embeddings
    embeddings = create_embeddings()

    # Create vector store
    vector_store = create_vector_store(text_chunks, embeddings)

    # Load LLM model
    llm_model_path = './LLM_model/Mistral/mistral-7b-instruct-v0.1.Q4_K_M'
    llm_model = create_llms_model(llm_model_path)

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm_model, chain_type='stuff', retriever=vector_store.as_retriever(search_kwargs={"k": 2}), memory=memory)

    # Define conversation function
    def conversation_chat(query, max_new_tokens=8192, temperature=0.01):
        result = chain({"question": query}, max_new_tokens=max_new_tokens, temperature=temperature)
        return result["answer"]

    print("Processing")
    # Perform conversation
    user_input = "What is a Proforma?"
    output = conversation_chat(user_input)
    print("Bot: ", output)

if __name__ == "__main__":
    main()
