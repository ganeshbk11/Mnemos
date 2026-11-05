Mnemos
Upload a big PDF manual and ask it questions in plain English. Instead of scrolling through 200 pages, you get a short answer with the page number it came from.

Built as a RAG (retrieval augmented generation) project. Everything runs in a Streamlit app.

How it works
You upload a PDF.
The PDF is read page by page and cut into small chunks (about 1000 characters each, with a little overlap so sentences don't get chopped at the edges).
Each chunk is turned into numbers (an embedding) and stored in a vector database that lives in memory.
When you ask a question, the app finds the 3 chunks closest in meaning to your question.
Those chunks, with their page numbers, go to the LLM along with your question. The LLM is told to answer only from them, and to say "I couldn't find that in the manual" if the answer isn't there.
Each person only ever searches their own upload. Nothing is saved to disk, so when you close the tab, the data is gone.

Stack
Streamlit for the UI
LangChain for the loader, splitter and prompt
PyPDFLoader to read the PDF
BAAI/bge-base-en-v1.5 (Hugging Face) for embeddings, runs locally and is free
Chroma as the in-memory vector store
Groq (openai/gpt-oss-120b) as the LLM
Setup
Clone the repo and install the packages:

git clone https://github.com/ganeshbk11/Mnemos.git
cd Mnemos
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
Make a file called .env in the project folder and add your Groq key:

GROQ_API_KEY=your_key_here
Then run it:

streamlit run app.py
The first run downloads the embedding model (a few hundred MB), so it takes a minute. After that it loads from your disk.

Project files
ingest.py loads the PDF, splits it, embeds it and builds the vector store
main.py takes a question, searches the store, builds the prompt and asks the LLM
app.py is the Streamlit screen: upload button and chat
You can test the question part without the UI by running python main.py. It expects a Systemdesign.pdf in the folder, so change the file name in the test block at the bottom to any PDF you have.
