"""Small runtime translation helper for the tkinter GUI."""

from __future__ import annotations

import locale
from typing import Literal

Language = Literal["ja", "en"]

_language: Language = "en"


def resolve_language(language: str) -> Language:
    """Resolve ``auto`` and unsupported values to a supported GUI language."""
    if language in {"ja", "en"}:
        return language  # type: ignore[return-value]
    locale_name = locale.getlocale()[0] or ""
    return "ja" if locale_name.lower().startswith("ja") else "en"


def set_language(language: str) -> Language:
    """Set and return the active GUI language."""
    global _language
    _language = resolve_language(language)
    return _language


def get_language() -> Language:
    return _language


def tr(ja: str, en: str) -> str:
    """Return Japanese or English text for the active language."""
    return ja if _language == "ja" else en
