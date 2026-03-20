import os
import csv
import time
import string
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import adversarial_dataset

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    model_kwargs={"parallel_tool_calls": False}
)

def normalize_answer(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

results = []

print("\n🚀 Starting CoT (No Tools) Evaluation Pipeline...\n")

for i, data in enumerate(adversarial_dataset):
    question = data["question"]
    ground_truth = data["answer"]

    print(f"\n--- Question {i+1}/{len(adversarial_dataset)} ---")
    print(f"Q: {question}")

    predicted_answer = "AGENT FAILED"
    error_status = "None"

    try:
        response = llm.invoke(
            f"Think step-by-step to answer this question. "
            f"If the answer does not exist, reply 'unavailable'. Question: {question}"
        )
        predicted_answer = response.content
        print(f"💭 Thought/Answer: {predicted_answer}")

    except Exception as e:
        error_status = str(e)
        print(f"❌ Error: {error_status}")

    norm_truth = normalize_answer(ground_truth)
    norm_pred = normalize_answer(predicted_answer)

    truth_words = set(norm_truth.split())
    pred_words = set(norm_pred.split())
    exact_match = 1 if truth_words.issubset(pred_words) and predicted_answer != "AGENT FAILED" else 0

    print(f"\nGround Truth: {ground_truth}")
    print(f"EM Score: {exact_match}")

    results.append({
        "Question": question,
        "Ground Truth": ground_truth,
        "Predicted": predicted_answer,
        "Exact Match": exact_match,
        "Error": error_status
    })
    
    time.sleep(2)

csv_filename = os.path.join(os.path.dirname(__file__), "..", "results", "cot_results.csv")

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["Question", "Ground Truth", "Predicted", "Exact Match", "Error"]
    )
    writer.writeheader()
    writer.writerows(results)

total = len(results)
correct = sum(r["Exact Match"] for r in results)
accuracy = correct / total

print("\n✅ Evaluation Complete!")
print(f"📊 CoT Accuracy: {accuracy:.2f}")
print(f"📁 Results saved to {csv_filename}")