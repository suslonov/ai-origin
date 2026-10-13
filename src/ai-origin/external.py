"""Prepare manual coder handoffs and import the resulting prompt once.

No CLI, model, or external process is started here. A handoff is a private bundle
of immutable experiment evidence and one writable TXT output. Importing freezes
that TXT into the usual candidate JSON so later runs keep reproducible snapshots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from graphs import improvement_instructions, improvement_rows, improvement_content
from schemas import CandidateInput, ImprovementInput
from storage import new_id, now

if TYPE_CHECKING:
    from service import Service


def prepare_handoff(service: Service, config: ImprovementInput, run: dict) -> dict:
    rows = improvement_rows(run, config.sample_scope)
    job_id, candidate_id = new_id("improvement"), new_id("candidate")
    folder = (service.store.root / "handoffs" / job_id).resolve()
    with service.store.lock:
        folder.mkdir(parents=True, mode=0o700)
        (folder / "samples").mkdir(mode=0o700)
        sample_paths = []
        for row in rows:
            path = folder / "samples" / (row["sample"]["id"] + ".json")
            service.store.atomic(path, json.dumps(row["sample"], ensure_ascii=False, indent=2))
            sample_paths.append(str(path))
        parent = folder / "parent-prompt.txt"
        evidence = folder / "evidence.json"
        instructions = folder / "instructions.md"
        output = service.store.path("candidates", candidate_id, ".txt").resolve()
        service.store.atomic(parent, run["prompt"]["text"])
        service.store.atomic(evidence, improvement_content(run, config))
        coder = "Claude Code" if config.runner == "claude_code" else "Codex"
        brief = f"Read {instructions} and follow it. Write the complete improved prompt to {output}."
        document = f"""# AI-origin prompt improvement for {coder}

{improvement_instructions(config.guidance, external=True)}

## Files and exact source

Source run ID: `{run['id']}`. Candidate ID: `{candidate_id}`.
Sample scope: `{config.sample_scope}`. {len(rows)} successful sample results are included.
These copies are frozen from the selected run, including its labels and materials;
current library files may have changed. Use only this bundle as experiment evidence.

- Parent prompt (exact version evaluated): `{parent}`
- Evidence, model results/explanations, evaluation settings and selected metrics: `{evidence}`
- Output file (already registered in the dashboard): `{output}`

### Sample snapshot files

""" + "\n".join(f"- `{path}`" for path in sample_paths) + f"""

## Deliverable

Read every listed sample and the evidence file. Produce one focused, generalizable
prompt revision using the reviewer guidance above as the primary objective.
Write the COMPLETE replacement system prompt as UTF-8 plain text to `{output}`.
No JSON wrapper, Markdown code fences, diff, placeholder, or commentary in that file.
It must be nonempty and at most 100,000 characters. Write a temporary sibling file
and atomically rename it over the output only after the complete prompt is ready.
The dashboard may import it when the user opens the candidate or starts a run.
Do not edit metadata JSON, source samples, other versions, or repository files.
Do not call paid model APIs or run an evaluation; the user will backtest separately.
Briefly explain your changes in your final reply, not in the output file.
"""
        service.store.atomic(instructions, document)
        # Empty output is intentional: the version is selectable immediately,
        # but prompt() refuses evaluation until a valid nonempty output exists.
        service.store.text("candidates", candidate_id, "")
        candidate = {"id": candidate_id, "name": config.name, "prompt": "", "rationale": f"Prepared for {coder}.",
                     "parent_id": run["prompt"]["id"], "parent_prompt": run["prompt"]["text"],
                     "sha256": "", "status": "awaiting_external", "created_at": now(),
                     "improvement_id": job_id, "reviews": [], "external": {"runner": config.runner,
                     "instructions_path": str(instructions), "output_path": str(output), "launch_prompt": brief}}
        job = {"id": job_id, "created_at": now(), "status": "awaiting_external", "config": config.model_dump(),
               "candidate_id": candidate_id, "sample_ids": [r["sample"]["id"] for r in rows],
               "instructions_path": str(instructions), "output_path": str(output), "launch_prompt": brief}
        service.store.write("candidates", candidate)
        service.store.write("improvements", job)
        return job


def import_external(service: Service, candidate_id: str) -> dict:
    from service import digest
    with service.store.lock:
        candidate = service.store.get("candidates", candidate_id)
        if candidate["status"] != "awaiting_external":
            return candidate
        # Derive the path from the validated record ID, never trust an arbitrary
        # path supplied in metadata or an HTTP request.
        path = service.store.path("candidates", candidate_id, ".txt")
        if not path.exists():
            return candidate
        if path.stat().st_size > 400_000:
            raise ValueError("External prompt is too large (maximum 100,000 characters).")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return candidate
        validated = CandidateInput(name=candidate["name"], prompt=text, parent_id=candidate["parent_id"])
        if text.startswith('```'):
            raise ValueError("External output must contain the plain prompt, without code fences.")
        candidate.update(prompt=validated.prompt, sha256=digest(validated.prompt), status="proposed", imported_at=now())
        candidate["rationale"] = "Imported from " + candidate["external"]["runner"] + " output."
        service.store.write("candidates", candidate)
        service.update_job("improvements", candidate["improvement_id"], {"status": "completed", "finished_at": now()})
        return candidate
