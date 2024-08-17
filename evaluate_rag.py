import json
from langchain.schema.document import Document
from create_database import split_documents, add_to_chroma
from query_data import query_rag
from sklearn.metrics import f1_score

SQUAD_PATH = "squad_dataset.json"

def main():
    # Create (or update) the data store.
    load_squad_to_chroma()

def load_squad_to_chroma():
    with open(SQUAD_PATH, 'r') as f:
        squad_data = json.load(f)

    documents = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            metadata = {"source": article['title']}
            documents.append(Document(page_content=context, metadata=metadata))

    chunks = split_documents(documents)
    add_to_chroma(chunks)

def evaluate_rag(squad_data):
    exact_matches = []
    f1_scores = []

    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                ground_truth = qa['answers']['text']
                
                generated_answer = query_rag(question)

                # Exact Match (EM)
                em = int(generated_answer.strip() in ground_truth)
                exact_matches.append(em)

                # F1 Score
                # Tokenize ground truth and generated answer for F1 calculation
                from nltk.tokenize import word_tokenize
                ground_truth_tokens = word_tokenize(' '.join(ground_truth))
                generated_answer_tokens = word_tokenize(generated_answer)
                f1 = f1_score(ground_truth_tokens, generated_answer_tokens, average='micro')
                f1_scores.append(f1)

    # Compute average metrics
    average_em = sum(exact_matches) / len(exact_matches)
    average_f1 = sum(f1_scores) / len(f1_scores)

    print(f"Average Exact Match (EM): {average_em * 100:.2f}%")
    print(f"Average F1 Score: {average_f1:.2f}")

if __name__ == "__main__":
    main()