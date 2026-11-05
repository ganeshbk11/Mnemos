import uuid
from langchain_community.document_loaders  import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


eembeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def ingest(path):
    loader = PyPDFLoader(path) 
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    vector_store = Chroma(
        collection_name=str(uuid.uuid4()),
        embedding_function = eembeddings,
    )
    vector_store.add_documents(texts)
    
    return vector_store