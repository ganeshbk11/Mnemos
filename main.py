from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders  import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader = PyPDFLoader("Systemdesign.pdf") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)



eembeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vector_store = Chroma(
    collection_name="example",
    embedding_function = eembeddings,
    persist_directory="chromadb"
)

# vector_store.add_documents(texts)
# Remove-Item -Recurse -Force chromadb

# results = vector_store.similarity_search({question}, k=3)
# for r in results:
#     print(r.metadata["page"], r.page_content)
#     print("*************")
question = input("Ask something: ")

results = vector_store.similarity_search(question, k=3)
    
context = "\n\n".join(r.page_content for r in results)

llm = init_chat_model(
    model = "openai/gpt-oss-120b",
    model_provider="groq"
    )

template = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a helpful assistant that answers questions using only the manual excerpts provided to you.

Rules:
- Use only the information in the excerpts. Do not use outside knowledge.
- If the answer is not in the excerpts, say "I couldn't find that in the manual." Do not guess.
- Keep answers short and clear. Use steps if the manual describes a procedure.
- Mention the page number the answer came from, like "(page 9)".
"""),
        (
            "human","""context ={context} 
            question = {question} """
        )
    ]
)

prompt = template.invoke({"context":context ,
                          "question":question})

answer = llm.invoke(prompt)

print(answer.content[:500])
