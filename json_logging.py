"""JSON logging helpers for the prediction API."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "event": "prediction",
        }

        status = getattr(record, "prediction_status", None)
        if status is not None:
            event["status"] = str(status)

        version = getattr(record, "model_version", None)
        if version is not None:
            event["model_version"] = str(version)

        error_type = getattr(record, "error_type", None)
        if error_type is not None:
            event["error_type"] = str(error_type)

        return json.dumps(event, separators=(",", ":"), ensure_ascii=False)


def configure_json_logging(
    logger_name: str = "mp272.predictions",
    level: int = logging.INFO,
) -> logging.Logger:

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Replace handlers so repeated app startup in tests/reloads does not emit
    # duplicate prediction lines.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_prediction(
    logger: logging.Logger,
    *,
    status: str,
    model_version: str | int | None = None,
    error: BaseException | None = None,
) -> None:
  

    extra: dict[str, str] = {"prediction_status": status}
    if model_version is not None:
        extra["model_version"] = str(model_version)
    if error is not None:
        extra["error_type"] = type(error).__name__

    if error is None:
        logger.info("prediction", extra=extra)
    else:
        logger.error("prediction", extra=extra)


def add_prediction_request_logging(
    app: Any,
    *,
    model_version: str | int | None = None,
    logger: logging.Logger | None = None,
) -> None:
  

    prediction_logger = logger or configure_json_logging()

    @app.middleware("http")
    async def prediction_logging_middleware(request: Any, call_next: Any) -> Any:
        if request.method != "POST" or request.url.path != "/predict":
            return await call_next(request)

        try:
            response = await call_next(request)
        except Exception as exc:
            log_prediction(
                prediction_logger,
                status="error",
                model_version=model_version,
                error=exc,
            )
            raise

        status = "success" if response.status_code < 400 else "error"
        extra: dict[str, str] = {"prediction_status": status}
        if model_version is not None:
            extra["model_version"] = str(model_version)
        if status == "success":
            prediction_logger.info("prediction", extra=extra)
        else:
            prediction_logger.error("prediction", extra=extra)
        return response

