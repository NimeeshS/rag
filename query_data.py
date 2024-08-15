import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE_ANSWER = """
Answer the question based only on the following context:

{context}

---

Answer the question based ONLY on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = [doc.page_content for doc, _score in results]
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    prompt_template_answer = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_ANSWER)
    prompt_answer = prompt_template_answer.format(context="\n".join([f"({source}) {context}" for source, context in zip(sources, context_text)]), 
                                    question=query_text)

    model = Ollama(model="mistral")
    response_text_answer = model.invoke(prompt_answer)

    formatted_response = f"Response: \n{response_text_answer}\n\nSources: {sources}"
    print(formatted_response)
    return response_text_answer

if __name__ == "__main__":
    main()