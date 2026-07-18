from unittest.mock import patch

from ptyolox_garage.i18n import get_language, resolve_language, set_language, tr


def test_explicit_languages() -> None:
    set_language("ja")
    assert get_language() == "ja"
    assert tr("学習", "Train") == "学習"

    set_language("en")
    assert get_language() == "en"
    assert tr("学習", "Train") == "Train"


def test_auto_language_uses_locale() -> None:
    with patch("ptyolox_garage.i18n.locale.getlocale", return_value=("ja_JP", "UTF-8")):
        assert resolve_language("auto") == "ja"

    with patch("ptyolox_garage.i18n.locale.getlocale", return_value=("en_US", "UTF-8")):
        assert resolve_language("auto") == "en"


def test_unknown_language_falls_back_to_locale() -> None:
    with patch("ptyolox_garage.i18n.locale.getlocale", return_value=(None, None)):
        assert resolve_language("unknown") == "en"
