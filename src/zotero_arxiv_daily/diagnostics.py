"""Safe, bounded diagnostics for inclusion in status emails."""

from __future__ import annotations

import re

from omegaconf import OmegaConf


class RunDiagnostics:
    """Collect warning/error log records without leaking configured secrets."""

    def __init__(self, config, *, max_entries: int = 10, max_message_length: int = 400):
        self.max_entries = max_entries
        self.max_message_length = max_message_length
        self.entries: list[tuple[str, str]] = []
        self.dropped_count = 0
        self._seen: set[tuple[str, str]] = set()
        self._secrets = self._configured_secrets(config)

    @staticmethod
    def _configured_secrets(config) -> list[str]:
        paths = (
            "zotero.api_key",
            "llm.api.key",
            "reranker.api.key",
            "email.sender_password",
        )
        secrets = []
        for path in paths:
            value = OmegaConf.select(config, path)
            if value is not None:
                value = str(value)
                if value and value != "???":
                    secrets.append(value)
        return sorted(set(secrets), key=len, reverse=True)

    def _sanitize(self, message: str) -> str:
        sanitized = message
        for secret in self._secrets:
            sanitized = sanitized.replace(secret, "<redacted>")
        sanitized = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "<redacted>", sanitized)
        sanitized = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
            "Bearer <redacted>",
            sanitized,
        )
        sanitized = " ".join(sanitized.split())
        if len(sanitized) > self.max_message_length:
            sanitized = sanitized[: self.max_message_length - 3] + "..."
        return sanitized

    def add(self, level: str, message: str) -> None:
        entry = (level.upper(), self._sanitize(message))
        if entry in self._seen:
            return
        self._seen.add(entry)
        if len(self.entries) >= self.max_entries:
            self.dropped_count += 1
            return
        self.entries.append(entry)

    def __call__(self, message) -> None:
        record = message.record
        self.add(record["level"].name, record["message"])

    @property
    def has_errors(self) -> bool:
        return any(level == "ERROR" for level, _ in self.entries)
