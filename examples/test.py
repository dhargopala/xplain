from xplain.executor import XPLAINMetricCalculator

PROJECT_ID="project-id"
LOCATION="us-central1"
LLM_MODEL_NAME="gemini-2.5-flash"
EMBEDDER_MODEL_NAME="text-embedding-004"

llm_args = {
    "PROJECT_ID": PROJECT_ID,
    "LOCATION": LOCATION,
    "MODEL_NAME": LLM_MODEL_NAME,
    "output_tokens": 65535,   
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0
}

embedder_args = {
    "PROJECT_ID": PROJECT_ID,
    "LOCATION": LOCATION,
    "MODEL_NAME": EMBEDDER_MODEL_NAME
}

xplain_calculator = XPLAINMetricCalculator()

xplain_calculator.select_llm("VERTEX_AI", **llm_args)
xplain_calculator.select_embedder("VERTEX_AI", **embedder_args)

prompt = "Why do we need sleep?"
scores = xplain_calculator.compute_score(prompt)

print(f"XPLAIN Scores for the prompt: '{prompt}' :")
print("-" * 25)
for word, score in scores:
  print(f"{word:<10} | Score: {score:.4f}")
print("-" * 25)