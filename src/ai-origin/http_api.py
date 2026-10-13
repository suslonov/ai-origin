"""Mount the private experiment API inside the existing HTML dev server.

The rest of the dev site keeps its original behavior. This route accepts local
requests only unless AI_ORIGIN_TOKEN is configured; remote users enter that
token in the dashboard. A custom header and same-origin check prevent other
websites from starting paid jobs through the user's browser.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError

from provider import ProviderError
from schemas import CandidateInput, CandidateRename, CandidateReview, ImprovementInput, RunInput, SampleInput
from service import Service

LOG = logging.getLogger(__name__)
_service: Service | None = None
_lock = threading.Lock()


def get_service() -> Service:
    global _service
    with _lock:
        if _service is None:
            _service = Service()
        return _service


def authorize(handler: BaseHTTPRequestHandler) -> None:
    token = os.environ.get("AI_ORIGIN_TOKEN", "")
    if token:
        supplied = handler.headers.get("Authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, token):
            raise PermissionError("Enter the server's AI_ORIGIN_TOKEN to access the dashboard.")
    else:
        peer = ipaddress.ip_address(handler.client_address[0])
        host = urlsplit("//" + handler.headers.get("Host", "")).hostname
        if not peer.is_loopback or host not in {"localhost", "127.0.0.1", "::1"}:
            raise PermissionError("Local access only. Set AI_ORIGIN_TOKEN for authenticated remote access.")
    if handler.command != "GET":
        if handler.headers.get("X-AI-Origin") != "1":
            raise PermissionError("Missing dashboard request header.")
        origin = handler.headers.get("Origin")
        if origin and urlsplit(origin).netloc != handler.headers.get("Host"):
            raise PermissionError("Cross-origin writes are not allowed.")


def send(handler: BaseHTTPRequestHandler, value: object, status: int = 200,
         mime: str = "application/json; charset=utf-8", filename: str | None = None) -> None:
    body = (json.dumps(value, ensure_ascii=False, allow_nan=False) if mime.startswith("application/json") else str(value)).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    if filename:
        handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.end_headers()
    handler.wfile.write(body)


def handle(handler: BaseHTTPRequestHandler) -> None:
    try:
        authorize(handler)
        service = get_service()
        parsed = urlsplit(handler.path)
        parts = parsed.path.removeprefix("/ai-origin/api/").strip("/").split("/")
        query = parse_qs(parsed.query)
        if handler.command == "GET":
            if parts == ["config"]:
                return send(handler, service.config())
            if parts == ["models"]:
                return send(handler, {"models": service.provider.models(refresh=query.get("refresh") == ["1"])})
            if parts == ["samples"]:
                # Scan current JSON files for each search; this small library
                # needs no index. Keep large text fields out of list responses.
                search = query.get("q", [""])[0].strip().casefold()
                samples = service.store.list("samples")
                if search:
                    samples = [s for s in samples if any(
                        search in str(s.get(field, "")).casefold()
                        for field in ("title", "id", "text"))]
                return send(handler, [{k: v for k, v in s.items() if k not in {"text", "materials", "assignment"}}
                                      for s in samples])
            if parts == ["prompt"]:
                return send(handler, service.prompt(query.get("id", ["repo"])[0]))
            if len(parts) == 1 and parts[0] in {"runs", "candidates", "improvements"}:
                return send(handler, service.history(parts[0]))
            if len(parts) == 2 and parts[0] in {"samples", "runs", "candidates", "improvements"}:
                detail = service.candidate_detail(parts[1]) if parts[0] == "candidates" else service.store.get(*parts)
                return send(handler, detail)
            if len(parts) == 3 and parts[0] == "runs" and parts[2] == "csv":
                return send(handler, service.csv_results(parts[1]), mime="text/csv; charset=utf-8", filename=parts[1] + ".csv")
            if len(parts) == 3 and parts[0] == "candidates" and parts[2] == "txt":
                return send(handler, service.prompt(parts[1])["text"], mime="text/plain; charset=utf-8", filename=parts[1] + ".txt")
        elif handler.command == "POST":
            if handler.headers.get_content_type() != "application/json":
                raise ValueError("Use application/json.")
            length = int(handler.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("Request body must be between 1 byte and 2 MB.")
            body = json.loads(handler.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            if parts == ["samples"]:
                return send(handler, service.save_sample(SampleInput.model_validate(body)), 201)
            if len(parts) == 2 and parts[0] == "samples":
                revision = body.pop("revision", None)
                return send(handler, service.save_sample(SampleInput.model_validate(body), parts[1], revision))
            if parts == ["runs"]:
                return send(handler, service.start_run(RunInput.model_validate(body)), 202)
            if len(parts) == 3 and parts[0] == "runs" and parts[2] == "cancel":
                return send(handler, service.cancel_run(parts[1]))
            if parts == ["improvements"]:
                return send(handler, service.start_improvement(ImprovementInput.model_validate(body)), 202)
            if parts == ["candidates"]:
                return send(handler, service.save_candidate(CandidateInput.model_validate(body)), 201)
            if len(parts) == 3 and parts[0] == "candidates" and parts[2] == "rename":
                return send(handler, service.rename_candidate(parts[1], CandidateRename.model_validate(body)))
            if len(parts) == 3 and parts[0] == "candidates" and parts[2] == "delete":
                service.delete_candidate(parts[1])
                return send(handler, {"id": parts[1], "deleted": True})
            if len(parts) == 3 and parts[0] == "candidates" and parts[2] == "review":
                return send(handler, service.review_candidate(parts[1], CandidateReview.model_validate(body)))
        send(handler, {"error": "Route not found"}, 404)
    except PermissionError as exc:
        send(handler, {"error": str(exc)}, 403)
    except FileNotFoundError:
        send(handler, {"error": "Record not found"}, 404)
    except ValidationError as exc:
        errors = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
        send(handler, {"error": "Check the form fields.", "details": errors}, 400)
    except (ValueError, ProviderError) as exc:
        send(handler, {"error": str(exc)}, 400)
    except Exception:
        LOG.exception("AI-origin API error")
        send(handler, {"error": "Server error. Check the dev-server log and data directory permissions."}, 500)
