import asyncio
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

async def main():
    # Load the Amnesty QA dataset
    questions = amnesty_qa['eval']['question']
    contexts = amnesty_qa['eval']['contexts']
    answers = amnesty_qa['eval']['answer']

    # Process the dataset into documents
    documents = create_documents(questions, contexts, answers)

    # Add to Chroma
    add_to_chroma_from_dataset(documents)

    # Run evaluation
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
            "answer": answer,
            "source": f"amnesty_qa:{i}"  # Example source ID
        }

        # Create a Document object for each context
        documents.append(Document(page_content=context[0], metadata=metadata))

    return documents

def add_to_chroma_from_dataset(documents: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Generate IDs and check against existing ones in Chroma
    chunks_with_ids = calculate_chunk_ids(documents)
    existing_items = db.get(include=[])  # Fetch existing documents
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter out documents that are already in the database
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        for chunk in tqdm(new_chunks, desc="👉 Adding new documents", unit="chunk"):
            db.add_documents([chunk], ids=[chunk.metadata["id"]])
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        question = chunk.metadata.get("question")
        current_page_id = f"{source}:{question}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def evaluate_rag():
    model = Ollama(model="mistral")

    # Run the evaluation
    try:
        result = evaluate(
            amnesty_qa['eval'],
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
            llm=model,
            embeddings=get_embedding_function
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    return result

if __name__ == "__main__":
    asyncio.run(main())
