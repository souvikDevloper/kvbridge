from pytest import MonkeyPatch

from kvbridge.provenance import code_revision, package_versions


def test_explicit_code_revision_wins(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("KVBRIDGE_CODE_REVISION", "release-test-commit")
    assert code_revision() == "release-test-commit"


def test_package_versions_marks_missing_distribution() -> None:
    resolved = package_versions(("kvbridge-package-that-does-not-exist",))

    assert resolved == {"kvbridge-package-that-does-not-exist": "not-installed"}
