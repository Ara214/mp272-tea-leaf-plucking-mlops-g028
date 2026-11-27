import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format each log record as a single JSON line."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "application_event"),
            "message": record.getMessage(),
        }

        for field in ("model_version", "status", "duration_ms", "error"):
            value = getattr(record, field, None)
            if value is not None:
                log_data[field] = value

        return json.dumps(log_data)


def get_logger(name="mp272"):
    """Configure a JSON logger that writes to stdout."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


def log_prediction(logger, model_version, status, duration_ms=None):
    """Log the outcome of a prediction request."""

    logger.info(
        "Prediction request processed",
        extra={
            "event": "prediction",
            "model_version": model_version,
            "status": status,
            "duration_ms": duration_ms,
        },
    )


def log_prediction_error(logger, error, model_version=None):
    """Log prediction errors without exposing request data."""

    logger.error(
        "Prediction request failed",
        extra={
            "event": "prediction_error",
            "model_version": model_version,
            "status": "error",
            "error": str(error),
        },
    )
