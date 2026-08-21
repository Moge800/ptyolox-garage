"""Tests for the public package version."""

from importlib.metadata import PackageNotFoundError, version

import pytest

import ptyolox_garage


def test_public_version_matches_installed_distribution() -> None:
    assert ptyolox_garage.__version__ == version("ptyolox-garage")


def test_version_uses_source_tree_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_name: str) -> str:
        raise PackageNotFoundError("ptyolox-garage")

    monkeypatch.setattr(
        ptyolox_garage._metadata, "version", missing_distribution
    )

    assert ptyolox_garage._resolve_version() == "0.0.0+local"


def test_unexpected_version_error_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_metadata(_name: str) -> str:
        raise RuntimeError("metadata failure")

    monkeypatch.setattr(
        ptyolox_garage._metadata, "version", failed_metadata
    )

    with pytest.raises(RuntimeError, match="metadata failure"):
        ptyolox_garage._resolve_version()


def test_version_is_in_public_directory() -> None:
    assert "__version__" in dir(ptyolox_garage)
