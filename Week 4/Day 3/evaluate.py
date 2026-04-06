import requests
import time

API_BASE_URL = "http://localhost:8000"

# --- 1. THE KNOWLEDGE BASE (Seed Data) ---
TEST_DOCUMENTS = [
    {
        "source": "doc_biology_cells.txt",
        "text": "Mitochondria are often referred to as the powerhouses of the cell. They help turn the energy we take from food into energy that the cell can use."
    },
    {
        "source": "doc_physics_gravity.txt",
        "text": "Gravity is a fundamental interaction which causes mutual attraction between all things that have mass. The strength of gravity is inversely proportional to the square of the distance between the objects."
    },
    {
        "source": "doc_tech_embeddings.txt",
        "text": "Vector embeddings are numerical representations of data. In Natural Language Processing, sentences with similar meanings will have embeddings that are mathematically close to each other in a high-dimensional space."
    },
    {
        "source": "doc_history_rome.txt",
        "text": "Julius Caesar was a Roman general and statesman. He played a critical role in the events that led to the demise of the Roman Republic and the rise of the Roman Empire."
    }
]

# --- 2. THE EVALUATION DATASET (Ground Truth) ---
# We map queries to the exact document source we EXPECT the API to return.
EVAL_DATASET = [
    {
        "query": "What part of the cell produces energy?",
        "expected_source": "doc_biology_cells.txt"
    },
    {
        "query": "How do vector databases represent text meaning?",
        "expected_source": "doc_tech_embeddings.txt"
    },
    {
        "query": "Who caused the fall of the Roman Republic?",
        "expected_source": "doc_history_rome.txt"
    },
    {
        "query": "Does distance affect the strength of mutual attraction between masses?",
        "expected_source": "doc_physics_gravity.txt"
    }
]

# --- 3. EVALUATION LOGIC ---
def seed_database():
    print("Seeding the vector database with test documents...")
    response = requests.post(f"{API_BASE_URL}/api/seed", json={"documents": TEST_DOCUMENTS})
    response.raise_for_status()
    print(f"Server Response: {response.json()['message']}\n")

def calculate_mrr(retrieved_sources, expected_source):
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / rank
    return 0.0

def run_evaluation(top_k=3):
    total_queries = len(EVAL_DATASET)
    hits = 0
    total_mrr = 0.0

    print(f"Running Retrieval Evaluation (Top-{top_k})...")
    print("-" * 50)

    for item in EVAL_DATASET:
        query = item["query"]
        expected_source = item["expected_source"]
        
        # Call the search API
        response = requests.get(f"{API_BASE_URL}/api/search", params={"query": query, "top_k": top_k})
        results = response.json().get("results", [])
        
        # Extract sources from results
        retrieved_sources = [res["source"] for res in results]
        
        # Calculate Metrics
        is_hit = expected_source in retrieved_sources
        if is_hit:
            hits += 1
            
        rr = calculate_mrr(retrieved_sources, expected_source)
        total_mrr += rr

        print(f"Q: '{query}'")
        print(f"Expected: {expected_source}")
        print(f"Retrieved: {retrieved_sources}")
        print(f"Score -> Hit: {'✅' if is_hit else '❌'} | RR: {rr:.2f}\n")

    # Final Aggregation
    overall_hit_rate = (hits / total_queries) * 100
    mean_mrr = total_mrr / total_queries

    print("=" * 50)
    print("🎯 FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Hit Rate (Recall@{top_k}):    {overall_hit_rate:.1f}%")
    print(f"Mean Reciprocal Rank:    {mean_mrr:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    try:
        seed_database()
        time.sleep(1) # Give the server a brief moment
        run_evaluation(top_k=3)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the API. Make sure 'app.py' is running in another terminal.")