"""
measure_tokens.py — Token Usage Comparison: Week 7 vs Week 8 Day 1
===================================================================
Measures the estimated tokens_in for the SAME 5 representative queries
under BOTH the old (Week 7) prompt architecture and the new (Week 8) one.

Methodology:
  - Uses observability.estimate_tokens() (4 chars/token heuristic) — same
    function used at runtime so numbers are consistent with production logs.
  - Does NOT call the LLM — purely measures prompt sizes.
  - Week 7 numbers are reconstructed from the original prompt templates.
  - Week 8 numbers use the new compressed templates + context truncation.
  - "Context" is simulated as a 2400-token block (≈ avg retrieved context
    from 3 chunks × 800 tokens each, representing a typical simple query).
  - History is simulated as 4 messages × 40 tokens each (=160 tokens).

Run from: Week 8/Day 1 - LLM Cost Optimization/
    python measure_tokens.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from observability import estimate_tokens

# ---------------------------------------------------------------------------
# Simulated context/history (representative values)
# ---------------------------------------------------------------------------
SAMPLE_CONTEXT_3_CHUNKS = (
    "[Source 1] (shipping_policy.txt)\n"
    + "A" * 800 * 4  # ~800 tokens × 4 chars/token
    + "\n\n[Source 2] (tracking.txt)\n"
    + "B" * 800 * 4
    + "\n\n[Source 3] (rates_and_zones.txt)\n"
    + "C" * 800 * 4
)

SAMPLE_HISTORY_6 = "User: Hello\nAgent: Hi!\n" * 6   # 6 message approximation
SAMPLE_HISTORY_4 = "User: Hello\nAgent: Hi!\n" * 4   # 4 message tighter window
SAMPLE_HISTORY_2 = "User: Hello\nAgent: Hi!\n" * 2   # 2 messages for answer

# 5 representative test queries
QUERIES = [
    "What is the standard delivery time for Zone 3?",
    "How do I file a claim for a lost shipment?",
    "What are the prohibited items for international shipping?",
    "Can you explain the difference between economy and express rates for heavy packages?",
    "What happens if my shipment is delayed at customs?",
]

# ---------------------------------------------------------------------------
# Week 7 Day 1 prompt templates (original, verbose)
# ---------------------------------------------------------------------------
def w7_intent_prompt(q):
    return (
        "You are an intent classifier for a logistics company's customer support chatbot.\n\n"
        "Classify the following user message into EXACTLY ONE of these intents:\n"
        "  - 'logistics'  : question about shipping, tracking, delivery, rates, zones, "
        "customs, SLAs, claims, prohibited items, packaging, returns, or any logistics topic.\n"
        "  - 'greeting'   : a greeting, farewell, thank you, or social small-talk.\n"
        "  - 'off_topic'  : anything completely unrelated to logistics.\n\n"
        f"User message: \"{q}\"\n\n"
        "Reply with only the intent label: 'logistics', 'greeting', or 'off_topic'."
    )

def w7_complexity_prompt(q):
    return (
        "Classify this logistics customer query as 'simple' or 'complex'.\n"
        "  'simple'  : single fact, one entity, one date, one status lookup.\n"
        "  'complex' : comparisons, multi-step, multi-zone, aggregations.\n\n"
        f"Query: {q}\n\nReply with only 'simple' or 'complex'."
    )

def w7_rephrase_prompt(q, history):
    return (
        "You are a query reformulation assistant for a logistics customer support system.\n\n"
        "Given the conversation history and the latest user question, rewrite the question "
        "as a STANDALONE question that can be understood without the history.\n\n"
        "Rules:\n"
        "- Replace pronouns (it, they, that, this) with the explicit entity they refer to.\n"
        "- Keep the question concise and preserve the original intent.\n"
        "- If the question is already standalone, return it unchanged.\n\n"
        f"Conversation history:\n{history}\n"
        f"Latest question: {q}\n\n"
        "Standalone question:"
    )

def w7_grade_prompt(q, ctx):
    return (
        "You are a document relevance grader for a logistics support system.\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        f"Customer question: {q}\n\n"
        "Does the context contain information that can answer the customer's question? "
        "Reply with ONLY 'yes' or 'no'."
    )

def w7_answer_instruction(ctx, history):
    return (
        "You are a professional customer support agent for SwiftShip Logistics. "
        "Your role is to help customers with shipping, tracking, rates, customs, claims, and related topics.\n\n"
        "You have access to the following retrieved knowledge base extracts:\n\n"
        "=== KNOWLEDGE BASE ===\n"
        f"{ctx}\n"
        "======================\n\n"
        "Instructions:\n"
        "1. Answer ONLY based on the provided knowledge base. Do not use outside knowledge.\n"
        "2. Be professional, empathetic, and concise — you are talking to a customer.\n"
        "3. If the query is time-sensitive (delays, lost shipments), set needs_escalation=true.\n"
        "4. Provide confidence from 0.0 to 1.0 based on how well the context answers.\n"
        "5. Reference sources as [Source N] in your answer.\n"
        "6. If the knowledge base is insufficient, set needs_escalation=true and confidence < 0.5.\n"
        "7. Do NOT hallucinate tracking numbers, dates, or details not in the context.\n\n"
        "Answer the following customer question:"
    )

# ---------------------------------------------------------------------------
# Week 8 Day 1 prompt templates (optimized, compressed)
# ---------------------------------------------------------------------------
def w8_combined_prompt(q):
    return (
        "Classify this logistics support message.\n\n"
        "intent options: 'logistics' (shipping/tracking/rates/customs/claims), "
        "'greeting' (hello/bye/thanks), 'off_topic' (unrelated).\n"
        "query_type options: 'simple' (single fact lookup), 'complex' (multi-step/comparisons).\n\n"
        f"Message: \"{q}\"\n\nReply with intent and query_type only."
    )

def w8_rephrase_prompt(q, history):
    return (
        "Rewrite the latest question as a standalone question using the history.\n"
        "Replace pronouns with explicit entities; keep concise; if already standalone return unchanged.\n\n"
        f"History:\n{history}\nQuestion: {q}\n\nStandalone:"
    )

def w8_grade_prompt(q, ctx_truncated):
    return (
        f"Context:\n{ctx_truncated}\n\nQuestion: {q}\n\n"
        "Does this context answer the question? Reply ONLY 'yes' or 'no'."
    )

def w8_answer_instruction(ctx_truncated):
    return (
        "You are a SwiftShip Logistics customer support agent.\n"
        "Answer ONLY from the knowledge base below. Be concise and professional.\n"
        "Rules: cite sources as [Source N]; set needs_escalation=true for urgent issues "
        "(lost/delayed shipments) or insufficient context; confidence 0.0-1.0; "
        "do NOT hallucinate tracking numbers or dates not in context.\n\n"
        "=== KNOWLEDGE BASE ===\n"
        f"{ctx_truncated}\n"
        "=====================\n\nCustomer question:"
    )

# ---------------------------------------------------------------------------
# Simulate context truncation (MAX_CONTEXT_TOKENS=2000)
# ---------------------------------------------------------------------------
MAX_CTX = 2000
def truncate_ctx(ctx, max_tokens=MAX_CTX):
    """Simulate _truncate_to_budget — take first max_tokens worth of context."""
    chars = max_tokens * 4
    return ctx[:chars] if len(ctx) > chars else ctx

FULL_CTX     = SAMPLE_CONTEXT_3_CHUNKS
TRUNC_CTX    = truncate_ctx(FULL_CTX)

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
print("=" * 72)
print("TOKEN USAGE COMPARISON: Week 7 Day 1  vs  Week 8 Day 1")
print("(estimates using 4 chars/token heuristic — same as production)")
print("=" * 72)
print(f"\n{'Query':<55} {'W7 in':>7} {'W8 in':>7} {'Saving':>8} {'%':>6}")
print("-" * 83)

total_w7, total_w8 = 0, 0

for q in QUERIES:
    short_q = q[:52] + "..." if len(q) > 52 else q

    # Week 7: 4 separate calls (intent + complexity + rephrase + grade + answer)
    # Rephrase uses history[-6:]; answer uses history[-4:]; grade uses full ctx
    w7_in = (
        estimate_tokens(w7_intent_prompt(q))
        + estimate_tokens(w7_complexity_prompt(q))
        + estimate_tokens(w7_rephrase_prompt(q, SAMPLE_HISTORY_6))
        + estimate_tokens(w7_grade_prompt(q, FULL_CTX))
        + estimate_tokens(w7_answer_instruction(FULL_CTX, SAMPLE_HISTORY_4) + "\n\n" + q)
    )

    # Week 8: merged classifier + rephrase (skipped for no-pronoun queries) + grade + answer
    # Determine if rephrase is skipped (no pronouns in query)
    import re
    PRONOUN_RE = re.compile(
        r"\b(it|its|they|them|their|that|this|those|these|he|she|him|her|his|hers)\b",
        re.IGNORECASE,
    )
    rephrase_skipped = not bool(PRONOUN_RE.search(q))

    w8_in = estimate_tokens(w8_combined_prompt(q))
    if not rephrase_skipped:
        w8_in += estimate_tokens(w8_rephrase_prompt(q, SAMPLE_HISTORY_4))
    w8_in += estimate_tokens(w8_grade_prompt(q, TRUNC_CTX))
    w8_in += estimate_tokens(w8_answer_instruction(TRUNC_CTX) + "\n\n" + q)

    saving = w7_in - w8_in
    pct    = round(saving / w7_in * 100, 1) if w7_in else 0
    rephrase_note = " [skip]" if rephrase_skipped else ""

    print(f"{short_q:<55} {w7_in:>7,} {w8_in:>7,} {saving:>+8,} {pct:>5.1f}%{rephrase_note}")
    total_w7 += w7_in
    total_w8 += w8_in

total_saving = total_w7 - total_w8
total_pct    = round(total_saving / total_w7 * 100, 1) if total_w7 else 0

print("-" * 83)
print(f"{'TOTAL (5 queries)':<55} {total_w7:>7,} {total_w8:>7,} {total_saving:>+8,} {total_pct:>5.1f}%")
print(f"{'AVERAGE per query':<55} {total_w7//5:>7,} {total_w8//5:>7,} {total_saving//5:>+8,} {total_pct:>5.1f}%")
print("=" * 72)
print("\nContext: 3-chunk retrieval (~2400 token raw ctx), 4-msg history")
print(f"Context after W8 truncation: ~{estimate_tokens(TRUNC_CTX):,} tokens (cap={MAX_CTX})")
print("\nNodes  W7 Day 1          ->  W8 Day 1")
print("  intent_classifier      ->  [MERGED into classify_intent_and_complexity]")
print("  classify_complexity    ->  [MERGED into classify_intent_and_complexity]")
print("  rephrase_query         ->  [SKIPPED if no pronouns]")
print("  grade_relevance        ->  [truncated context, compressed prompt]")
print("  generate_answer        ->  [truncated context, tighter history, compressed prompt]")
