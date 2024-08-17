import json
from create_database import split_documents, add_to_chroma, Document
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

print(amnesty_qa)

def evaluate_rag():
    model = Ollama(model="mistral")