"""Tests for bounded and sanitized run diagnostics."""

from zotero_arxiv_daily.diagnostics import RunDiagnostics


def test_diagnostics_deduplicate_bound_and_redact(config):
    diagnostics = RunDiagnostics(config, max_entries=2)
    warning = f"Retry with {config.zotero.api_key}"

    diagnostics.add("WARNING", warning)
    diagnostics.add("WARNING", warning)
    diagnostics.add("ERROR", "Request used Bearer top-secret-token")
    diagnostics.add("WARNING", "Third unique warning")

    assert diagnostics.entries == [
        ("WARNING", "Retry with <redacted>"),
        ("ERROR", "Request used Bearer <redacted>"),
    ]
    assert diagnostics.dropped_count == 1
    assert diagnostics.has_errors is True
