from pytest import MonkeyPatch

from kvbridge.provenance import code_revision


def test_explicit_code_revision_wins(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("KVBRIDGE_CODE_REVISION", "release-test-commit")
    assert code_revision() == "release-test-commit"
