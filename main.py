from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

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


def answer(question, vector_store):
    results = vector_store.similarity_search(question, k=3)
    # context = "\n\n".join(r.page_content for r in results)
    context = ""
    for r in results:
        page = r.metadata["page"] + 11
        context = context + "[page" + str(page) + "]\n" + r.page_content + "\n\n"


    prompt = template.invoke({"context":context ,
                            "question":question})

    return llm.invoke(prompt).content
