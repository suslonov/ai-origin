"""Application operations shared by the HTTP dashboard and the graph nodes.

One worker executes queued jobs sequentially. HTTP requests remain responsive,
and each completed sample is written immediately. The queue itself is reflected
in JSON history; after a restart unfinished jobs are marked interrupted instead
of automatically making more chargeable model requests.
"""
from __future__ import annotations

import copy
import csv
import difflib
import hashlib
import io
import os
import queue
import threading
import uuid
from pathlib import Path

from langsmith import tracing_context

from graphs import build_evaluation_graph, build_improvement_graph
from metrics import comparison, grouped_metrics, select_samples
from provider import OpenAIProvider
from schemas import CandidateInput, CandidateReview, ImprovementInput, RunInput, SampleInput
from storage import Store, new_id, now

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PROMPT = REPO_ROOT / "src/system_bucket/exam/grade/ai_origin.txt"
ACTIVE = {"queued", "running", "cancelling"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Service:
    def __init__(self, root: Path | None = None, provider: OpenAIProvider | None = None) -> None:
        # Preserve Anton's ~/ai-origin history; other cloners get their own home.
        self.store = Store(root or Path(os.environ.get("AI_ORIGIN_DATA_DIR", str(Path.home() / "ai-origin"))))
        self.provider = provider or OpenAIProvider()
        self.project = os.environ.get("LANGSMITH_PROJECT", "kvasir-ai-origin")
        # Missing tracing credentials do not stop local work. The dashboard
        # explicitly displays that traces are off rather than claiming upload.
        self.tracing = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")) and os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
        self.evaluation_graph = build_evaluation_graph(self)
        self.improvement_graph = build_improvement_graph(self)
        self.queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self.worker = threading.Thread(target=self._work, name="ai-origin-worker", daemon=True)
        self.worker.start()

    def config(self) -> dict:
        return {"data_dir": str(self.store.root), "openai_configured": self.provider.configured,
                "tracing_enabled": self.tracing, "langsmith_project": self.project,
                "repo_prompt": self.prompt("repo"), "graph_engine": "LangGraph"}

    def prompt(self, prompt_id: str) -> dict:
        if prompt_id == "repo":
            # Read on every new run, not once at server startup. A committed or
            # local edit to the actual evaluation prompt is used immediately.
            text = REPO_PROMPT.read_text(encoding="utf-8")
            return {"id": "repo", "name": "Current repository prompt", "text": text, "sha256": digest(text)}
        candidate = self.store.get("candidates", prompt_id)
        return {"id": candidate["id"], "name": candidate["name"], "text": candidate["prompt"], "sha256": candidate["sha256"]}

    def save_sample(self, data: SampleInput, sample_id: str | None = None, revision: int | None = None) -> dict:
        with self.store.lock:
            if sample_id:
                old = self.store.get("samples", sample_id)
                if revision != old["revision"]:
                    raise ValueError("This sample changed in another window. Reload it before saving.")
            else:
                old = {"id": new_id("sample"), "created_at": now(), "revision": 0}
            sample = {**data.model_dump(), "id": old["id"], "created_at": old["created_at"],
                      "updated_at": now(), "revision": old["revision"] + 1}
            self.store.write("samples", sample)
            return sample

    def save_candidate(self, data: CandidateInput, *, parent_text: str | None = None,
                       improvement_id: str | None = None) -> dict:
        parent = parent_text if parent_text is not None else self.prompt(data.parent_id)["text"]
        candidate = {**data.model_dump(), "id": new_id("candidate"), "created_at": now(),
                     "status": "proposed", "sha256": digest(data.prompt), "parent_prompt": parent,
                     "improvement_id": improvement_id, "reviews": []}
        # The JSON is authoritative; the sibling TXT is convenient for manual
        # diffing/copying. Editing in the dashboard creates a new version/ID.
        self.store.text("candidates", candidate["id"], data.prompt)
        self.store.write("candidates", candidate)
        return candidate

    def review_candidate(self, candidate_id: str, review: CandidateReview) -> dict:
        with self.store.lock:
            candidate = self.store.get("candidates", candidate_id)
            candidate["status"] = review.status
            candidate["reviews"].append({**review.model_dump(), "at": now()})
            self.store.write("candidates", candidate)
            return candidate

    def candidate_detail(self, candidate_id: str) -> dict:
        candidate = self.store.get("candidates", candidate_id)
        candidate["diff"] = "".join(difflib.unified_diff(
            candidate["parent_prompt"].splitlines(True), candidate["prompt"].splitlines(True),
            fromfile="Parent prompt", tofile=candidate["name"]))
        return candidate

    def start_run(self, config: RunInput) -> dict:
        self.provider.validate_model(config)
        prompt = self.prompt(config.prompt_id)
        if config.reference_run_id:
            reference = self.store.get("runs", config.reference_run_id)
            if reference["status"] in ACTIVE:
                raise ValueError("Wait for the reference run to finish before backtesting.")
            # Backtesting uses the old *snapshots*, not today's editable samples.
            samples = [copy.deepcopy(row["sample"]) for row in reference["results"]]
        else:
            samples = select_samples(self.store.list("samples"), config)
        if not samples:
            raise ValueError("No samples match this selection.")
        run = {"id": new_id("run"), "name": config.name or f"{config.model} · {prompt['name']}",
               "created_at": now(), "status": "queued", "config": config.model_dump(), "prompt": prompt,
               "cancel_requested": False, "trace_id": str(uuid.uuid4()),
               "results": [{"sample": s, "status": "pending"} for s in samples]}
        run["metrics"] = grouped_metrics(run["results"], config.threshold)
        self.store.text("runs", run["id"], prompt["text"])
        self.store.write("runs", run)
        self.queue.put(("runs", run["id"]))
        return run

    def start_improvement(self, config: ImprovementInput) -> dict:
        self.provider.validate_model(config)
        run = self.store.get("runs", config.run_id)
        if run["status"] in ACTIVE:
            raise ValueError("Wait for the source run to finish.")
        if not any(r["status"] == "completed" and r["sample"]["split"] == "development" for r in run["results"]):
            raise ValueError("Improvement needs completed development samples; holdout samples are never sent to the improver.")
        job = {"id": new_id("improvement"), "created_at": now(), "status": "queued",
               "config": config.model_dump(), "trace_id": str(uuid.uuid4())}
        self.store.write("improvements", job)
        self.queue.put(("improvements", job["id"]))
        return job

    def update_job(self, collection: str, job_id: str, changes: dict) -> None:
        with self.store.lock:
            item = self.store.get(collection, job_id)
            item.update(changes)
            self.store.write(collection, item)

    def update_row(self, run_id: str, index: int, changes: dict) -> None:
        with self.store.lock:
            run = self.store.get("runs", run_id)
            run["results"][index].update(changes)
            run["metrics"] = grouped_metrics(run["results"], run["config"]["threshold"])
            self.store.write("runs", run)

    def finish_run(self, run_id: str, fatal: bool, error: str) -> None:
        with self.store.lock:
            run = self.store.get("runs", run_id)
            failed = any(r["status"] == "failed" for r in run["results"])
            status = "failed" if fatal else ("completed_with_errors" if failed else "completed")
            if run["cancel_requested"]:
                status = "cancelled"
            run.update(status=status, finished_at=now())
            if fatal:
                run["error"] = error
            if run["config"]["reference_run_id"]:
                run["comparison"] = comparison(run["results"], self.store.get("runs", run["config"]["reference_run_id"]))
            self.store.write("runs", run)

    def cancel_run(self, run_id: str) -> dict:
        with self.store.lock:
            run = self.store.get("runs", run_id)
            if run["status"] in ACTIVE:
                run.update(cancel_requested=True, status="cancelling")
                self.store.write("runs", run)
            return run

    def history(self, collection: str) -> list[dict]:
        # History lists stay small. Full request/response and sample text are
        # loaded only when opening a particular run or sample.
        omit = {"results", "request", "raw_response", "prompt", "parent_prompt"}
        return [{k: v for k, v in item.items() if k not in omit} for item in self.store.list(collection)]

    def csv_results(self, run_id: str) -> str:
        run = self.store.get("runs", run_id)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["text_id", "text_link", "title", "lang", "manual_probability", "manual_low", "manual_high",
                         "origin", "materials_mode", "split", "model_probability", "status", "explanation", "error"])
        for row in run["results"]:
            s = row["sample"]
            cells = [s["id"], f"/ai-origin/?run={run_id}&sample={s['id']}", s["title"], s["lang"],
                     s["manual_probability"], s["manual_low"], s["manual_high"], s["origin"], s["materials_mode"], s["split"],
                     row.get("model_probability", ""), row["status"], row.get("explanation", ""), row.get("error", "")]
            # Quoting CSV alone does not neutralize spreadsheet formulas.
            writer.writerow(["'" + c if isinstance(c, str) and c.lstrip().startswith(("=", "+", "-", "@")) else c for c in cells])
        return output.getvalue()

    def _work(self) -> None:
        while True:
            queued = self.queue.get()
            if queued is None:
                self.queue.task_done()
                return
            collection, job_id = queued
            try:
                job = self.store.get(collection, job_id)
                self.update_job(collection, job_id, {"status": "running", "started_at": now(),
                                                     "tracing_enabled": self.tracing, "langsmith_project": self.project})
                graph = self.evaluation_graph if collection == "runs" else self.improvement_graph
                # LangGraph emits node traces automatically. The provider's
                # @traceable call becomes a child, with its exact input/output.
                with tracing_context(enabled=self.tracing, project_name=self.project):
                    graph.invoke({"job_id": job_id}, config={"run_id": uuid.UUID(job["trace_id"]),
                        "recursion_limit": max(50, len(job.get("results", [])) * 4 + 10),
                        "metadata": {"job_id": job_id, "kind": collection}, "tags": ["ai-origin", collection]})
            except Exception as exc:
                self.update_job(collection, job_id, {"status": "failed", "finished_at": now(), "error": str(exc)})
            finally:
                self.queue.task_done()

    def close(self) -> None:
        """For tests/graceful shutdown after jobs finish; no interrupted retries."""
        self.queue.put(None)
        self.worker.join(timeout=2)
        if not self.worker.is_alive():
            self.store.close()
