import argparse
from create_database import Document, Chroma, CHROMA_PATH, tqdm
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)
from ragas import evaluate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from datasets import load_dataset

amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2", trust_remote_code=True)

def main():
    # Load the Amnesty QA dataset
    questions = amnesty_qa['eval']['question']
    contexts = amnesty_qa['eval']['contexts']
    answers = amnesty_qa['eval']['answer']

    # Process the dataset into documents
    documents = create_documents(questions, contexts, answers)

    # Add to Chroma
    add_to_chroma_from_dataset(documents)

    result = evaluate_rag()

    print(result)

def create_documents(questions, contexts, answers):
    documents = []
    for i in range(len(questions)):
        context = contexts[i]
        question = questions[i]
        answer = answers[i]

        metadata = {
            "question": question,
            "answer": answer
        }

        # Create a Document object for each context
        documents.append(Document(page_content=context[0], metadata=metadata))

    return documents

def add_to_chroma_from_dataset(documents: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Add documents to the Chroma database
    for chunk in tqdm(documents, desc="👉 Adding new documents", unit="chunk"):
        db.add_documents([chunk])

def evaluate_rag():
    model = Ollama(model="mistral")

    result = evaluate(amnesty_qa, metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,], llm=model, embeddings=get_embedding_function)
    
    return result

if __name__ == "__main__":
    main()