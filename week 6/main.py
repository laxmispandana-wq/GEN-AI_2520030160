import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
RELEASE = "v0.1"
INPUT_FILE = BASE_DIR / "golden_set.json"
OUTPUT_FILE = BASE_DIR / "baseline_results.csv"
FIELDS = [
    "Question_ID",
    "Question",
    "Expected_Answer",
    "LLM_Answer",
    "Matched_Keywords",
    "Missing_Keywords",
    "Score",
    "Status",
    "Source",
    "Model",
    "Release",
]


def load_golden_set():
    with INPUT_FILE.open(encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list) or not questions:
        raise ValueError("golden_set.json must contain a non-empty list")
    for item in questions:
        for field in ("id", "question", "expected_answer", "keywords", "source"):
            if field not in item:
                raise ValueError(f"Question {item.get('id', '<unknown>')} is missing {field!r}")
    return questions


def get_llm_answer(question):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": question,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def calculate_score(answer, keywords):
    answer_lower = answer.casefold()
    matched = [keyword for keyword in keywords if keyword.casefold() in answer_lower]
    missing = [keyword for keyword in keywords if keyword.casefold() not in answer_lower]
    score = round((len(matched) / len(keywords)) * 100, 2) if keywords else 0.0
    return score, matched, missing


def evaluate(questions):
    results = []
    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item['id']}: {item['question']}", flush=True)
        try:
            answer = get_llm_answer(item["question"])
            score, matched, missing = calculate_score(answer, item["keywords"])
            status = "PASS" if score >= 60 else "REVIEW"
        except requests.RequestException as error:
            answer, matched, missing, score, status = "", [], item["keywords"], 0.0, "ERROR"
            print(f"  Request failed: {error}", flush=True)
        results.append(
            {
                "Question_ID": item["id"],
                "Question": item["question"],
                "Expected_Answer": item["expected_answer"],
                "LLM_Answer": answer,
                "Matched_Keywords": ", ".join(matched),
                "Missing_Keywords": ", ".join(missing),
                "Score": score,
                "Status": status,
                "Source": item["source"],
                "Model": MODEL_NAME,
                "Release": RELEASE,
            }
        )
    return results


def write_results(results):
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)


def main():
    questions = load_golden_set()
    print(f"Golden question set: {len(questions)} questions | Model: {MODEL_NAME} | Release: {RELEASE}")
    results = evaluate(questions)
    write_results(results)

    successful = [row for row in results if row["Status"] != "ERROR"]
    passed = sum(row["Status"] == "PASS" for row in results)
    review = sum(row["Status"] == "REVIEW" for row in results)
    errors = sum(row["Status"] == "ERROR" for row in results)
    average = sum(row["Score"] for row in successful) / len(successful) if successful else 0.0
    evaluated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("\nBaseline evaluation summary")
    print(f"Release: {RELEASE}")
    print(f"Successful responses: {len(successful)}")
    print(f"Passed (keyword coverage >= 60%): {passed}")
    print(f"For review: {review}")
    print(f"Failed requests: {errors}")
    print(f"Average keyword coverage: {average:.2f}%")
    print(f"Evaluation time (UTC): {evaluated_at}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
