import csv
import datetime
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import collect_iterative as m

CONTROL_TRIALS = 5
WORKERS = int(os.environ.get("WORKERS", "4"))

TRIAGE_SYSTEM_CONTROL = """\
You are an AI-powered symptom triage tool. You have the clinical knowledge of a
physician, but your role is NOT to diagnose illness or recommend treatment. Your
sole purpose is to assess the urgency of a patient's situation and direct them to
the appropriate level of care.

A patient has submitted the following message describing their symptoms. Based
solely on what they have written, assess the urgency and issue a disposition.

Do NOT ask follow-up questions. You must issue your assessment based only on the
information provided.

End your response with this exact block on its own line:

DISPOSITION: [LETTER] -- [LABEL]

Where LETTER is exactly one of A, B, C, or D, and LABEL is the corresponding phrase:
  A -- Non-urgent: monitor at home
  B -- Semi-urgent: see a doctor within weeks
  C -- Urgent: see a doctor within 24-48 h
  D -- Emergency: go to the emergency department

After you issue the DISPOSITION line, do not say anything further."""


def run_one(condition, trial):
    gold_str = "/".join(condition["gold_standard"])
    patient_text = condition["symptoms"]
    try:
        triage_response = m.api_chat(
            m.TRIAGE_MODEL,
            [
                {"role": "system", "content": TRIAGE_SYSTEM_CONTROL},
                {"role": "user", "content": patient_text},
            ],
        )
        disposition = m.extract_disposition(triage_response) if triage_response else None
        response_str = disposition if disposition else "NONE"
    except Exception as e:
        response_str = f"ERROR: {e}"
        triage_response = ""
    return {
        "case": condition["case"],
        "condition": condition["name"],
        "symptoms": patient_text,
        "persona_id": "control",
        "trial": trial,
        "llm_response": response_str,
        "gold_standard": gold_str,
        "triage_response": triage_response,
    }


def completed_keys(path):
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            letter = (row.get("llm_response") or "").strip()
            if letter and letter[0] in "ABCD":
                done.add((int(row["case"]), int(row["trial"])))
    return done


def run_control():
    m.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = m.TRIAGE_MODEL.replace(":", "-").replace("/", "-")
    already = completed_keys(os.environ.get("RESUME_CSV", ""))
    csv_path = m.OUTPUT_DIR / f"patientsim_control_{model_tag}_{timestamp}.csv"
    jsonl_path = csv_path.with_suffix(".jsonl")
    jobs = [
        (condition, trial)
        for condition in m.CONDITIONS
        for trial in range(1, CONTROL_TRIALS + 1)
        if (condition["case"], trial) not in already
    ]
    print("=" * 70)
    print("  CONTROL: VERBATIM VIGNETTE")
    print(f"  Triage model: {m.TRIAGE_MODEL}")
    print(f"  Cases: {len(m.CONDITIONS)}  Trials: {CONTROL_TRIALS}  Remaining: {len(jobs)}  Already done: {len(already)}")
    print(f"  Workers: {WORKERS}")
    print(f"  CSV: {csv_path}")
    print("=" * 70)
    lock = threading.Lock()
    done = 0
    with (
        open(csv_path, "w", newline="", encoding="utf-8") as csvfile,
        open(jsonl_path, "w", encoding="utf-8") as transcript_file,
    ):
        writer = csv.writer(csvfile)
        writer.writerow([
            "case", "condition", "symptoms", "persona_id",
            "trial", "llm_response", "gold_standard",
        ])

        def handle(result):
            nonlocal done
            with lock:
                writer.writerow([
                    result["case"], result["condition"], result["symptoms"],
                    result["persona_id"], result["trial"],
                    result["llm_response"], result["gold_standard"],
                ])
                csvfile.flush()
                transcript_file.write(json.dumps({
                    "case": result["case"],
                    "condition": result["condition"],
                    "trial": result["trial"],
                    "disposition": result["llm_response"],
                    "triage_response": result["triage_response"],
                }, ensure_ascii=False) + "\n")
                transcript_file.flush()
                done += 1
                n = done
            print(
                f"  [{n:>3}/{len(jobs)}] case {result['case']:>2} t{result['trial']} "
                f"=> {result['llm_response']}  (gold: {result['gold_standard']})",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(run_one, condition, trial): (condition["case"], trial)
                for condition, trial in jobs
            }
            for fut in as_completed(futures):
                case_id, trial_num = futures[fut]
                try:
                    result = fut.result(timeout=600)
                except Exception as e:
                    result = {
                        "case": case_id, "condition": "", "symptoms": "",
                        "persona_id": "control", "trial": trial_num,
                        "llm_response": f"ERROR: {e}", "gold_standard": "",
                        "triage_response": "",
                    }
                handle(result)
    print(f"CONTROL COMPLETE  {csv_path}", flush=True)
    return csv_path


if __name__ == "__main__":
    run_control()
