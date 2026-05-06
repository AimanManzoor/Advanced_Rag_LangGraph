"""Evaluation harness for the RAG graph.

Runs every question in eval/questions.jsonl through the graph and produces
a metrics report. Saves both a JSON results dump and a human-readable summary.
"""
import json
import time
from pathlib import Path
from statistics import median

from src.graph import app

EVAL_DIR = Path(__file__).parent
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_PATH = EVAL_DIR / "results.json"
SUMMARY_PATH = EVAL_DIR / "summary.md"


def load_questions():
    """Load the labeled question set."""
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def evaluate_one(q):
    """Run one question through the graph and capture metrics."""
    start = time.time()
    try:
        result = app.invoke({"query": q["query"], "attempts": 0})
        latency = time.time() - start
        error = None
    except Exception as e:
        return {
            "id": q["id"],
            "query": q["query"],
            "error": str(e),
            "latency": time.time() - start,
        }

    # Did the answer contain the expected substrings?
    answer = result.get("answer", "").lower()
    expected_substrings = q.get("answer_contains", [])
    answer_match = any(s.lower() in answer for s in expected_substrings) if expected_substrings else False

    # Did citations include any of the expected docs?
    citations = result.get("citations", [])
    expected_docs = q.get("expected_docs", [])
    citation_hit = any(d in citations for d in expected_docs) if expected_docs else len(citations) == 0

    # Was the gold doc in the reranked top-5?
    reranked_doc_ids = [d.metadata.get("doc_id") for d in result.get("reranked", [])]
    retrieval_at_5 = any(d in reranked_doc_ids for d in expected_docs) if expected_docs else True

    return {
        "id": q["id"],
        "query": q["query"],
        "expected_docs": expected_docs,
        "expected_substrings": expected_substrings,
        "answer": result.get("answer", ""),
        "citations": citations,
        "reranked_docs": reranked_doc_ids,
        "confidence": result.get("confidence", 0.0),
        "attempts": result.get("attempts", 0),
        "route": result.get("route", "?"),
        "rewritten": result.get("rewritten", ""),
        "answer_match": answer_match,
        "citation_hit": citation_hit,
        "retrieval_at_5": retrieval_at_5,
        "latency": latency,
        "error": None,
    }


def summarize(results):
    """Compute aggregate metrics."""
    successful = [r for r in results if r.get("error") is None]
    n = len(successful)
    if n == 0:
        return {"error": "all queries errored"}

    return {
        "total_questions": len(results),
        "successful_runs": n,
        "errored_runs": len(results) - n,
        "answer_match_rate": sum(r["answer_match"] for r in successful) / n,
        "citation_accuracy": sum(r["citation_hit"] for r in successful) / n,
        "retrieval_at_5": sum(r["retrieval_at_5"] for r in successful) / n,
        "avg_confidence": sum(r["confidence"] for r in successful) / n,
        "avg_attempts": sum(r["attempts"] for r in successful) / n,
        "median_latency_s": median(r["latency"] for r in successful),
        "p95_latency_s": sorted(r["latency"] for r in successful)[int(n * 0.95)] if n > 1 else successful[0]["latency"],
        "route_distribution": {
            "simple": sum(1 for r in successful if r["route"] == "simple"),
            "multi_hop": sum(1 for r in successful if r["route"] == "multi_hop"),
        },
    }


def write_summary(metrics, results):
    """Write a markdown summary for human review."""
    lines = [
        "# Eval Summary",
        "",
        f"- **Total questions:** {metrics['total_questions']}",
        f"- **Successful runs:** {metrics['successful_runs']}",
        f"- **Errors:** {metrics['errored_runs']}",
        f"- **Answer match rate:** {metrics['answer_match_rate']:.1%}",
        f"- **Citation accuracy:** {metrics['citation_accuracy']:.1%}",
        f"- **Retrieval@5:** {metrics['retrieval_at_5']:.1%}",
        f"- **Avg confidence:** {metrics['avg_confidence']:.2f}",
        f"- **Avg attempts:** {metrics['avg_attempts']:.2f}",
        f"- **Median latency:** {metrics['median_latency_s']:.2f}s",
        f"- **P95 latency:** {metrics['p95_latency_s']:.2f}s",
        f"- **Route distribution:** simple={metrics['route_distribution']['simple']}, multi_hop={metrics['route_distribution']['multi_hop']}",
        "",
        "## Failures",
        "",
    ]

    failures = [r for r in results if r.get("error") or not r.get("answer_match") or not r.get("citation_hit")]
    if not failures:
        lines.append("_No failures._")
    else:
        for r in failures:
            lines.append(f"### {r.get('id', '?')}: {r.get('query', '?')}")
            if r.get("error"):
                lines.append(f"- **ERRORED:** `{r['error']}`")
            lines.append(f"- Expected docs: `{r.get('expected_docs', [])}`")
            lines.append(f"- Cited: `{r.get('citations', [])}`")
            lines.append(f"- Reranked top-5: `{r.get('reranked_docs', [])}`")
            lines.append(f"- Confidence: {r.get('confidence', 0.0):.2f}")
            answer = r.get("answer", "(no answer)")
            lines.append(f"- Answer: {answer[:200]}...")
            lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines))

def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['query'][:60]}...", end=" ", flush=True)
        r = evaluate_one(q)
        if r.get("error"):
            print(f"ERROR: {r['error']}")
        else:
            marks = ("✓" if r["answer_match"] else "✗") + ("✓" if r["citation_hit"] else "✗")
            print(f"[{marks}] conf={r['confidence']:.2f} {r['latency']:.1f}s")
        results.append(r)

    metrics = summarize(results)

    RESULTS_PATH.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2))
    write_summary(metrics, results)

    print("\n" + "=" * 60)
    print("EVAL COMPLETE")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
