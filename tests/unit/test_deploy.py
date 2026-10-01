"""Docker build / test / push helpers and the M12 files (DOC-04 §11, §12, §16.5; DOC-05 M12).

Docker is never called here: a fake runner records the commands and returns canned results,
so the safety rules of ``make docker-push`` are proven without a registry. The real image is
built and tested by ``make docker-build`` / ``make docker-test`` and the CI ``docker`` job."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from house_price import deploy
from house_price.deploy import DeployError
from house_price.persistence import artifact
from tests.conftest import REPO_ROOT, artifact_metadata

LABEL = deploy.VERSION_LABEL
TEMPLATE = '{{ index .Config.Labels "' + LABEL + '" }}'


class FakeDocker:
    """Answers ``docker`` / ``git`` commands from a table of (prefix -> result)."""

    def __init__(self, answers: dict[tuple[str, ...], tuple[int, str, str]] | None = None):
        self.answers = answers or {}
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self.commands.append(command)
        for prefix, (code, out, err) in self.answers.items():
            if tuple(command[: len(prefix)]) == prefix:
                return subprocess.CompletedProcess(command, code, out, err)
        return subprocess.CompletedProcess(command, 0, "", "")

    def ran(self, *prefix: str) -> bool:
        return any(tuple(c[: len(prefix)]) == prefix for c in self.commands)


def write_artifact(root: Path, relative: str, **overrides: object) -> Path:
    directory = root / relative
    directory.mkdir(parents=True)
    sha = artifact.save_model({"stand-in": "model"}, directory)
    meta = artifact_metadata(model_sha256=sha, **overrides)
    artifact.write_metadata(meta, directory)  # type: ignore[arg-type]
    return Path(relative)


RELEASE = {"is_release": True, "model_version": "1.0.0", "mlflow": None}


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def release_artifact(root: Path) -> deploy.Artifact:
    from house_price.persistence import metadata

    links = metadata.MlflowLinks(refit="r1", final_holdout_evaluation="f1", release="rel-1")
    path = write_artifact(root, "models/1.0.0", **{**RELEASE, "mlflow": links})
    return deploy.load_artifact(path, "1.0.0")


# ------------------------------------------------------------------- artifact choice


def test_default_model_dirs() -> None:
    assert deploy.model_dir_for(None, None) == Path("artifacts/smoke/model")
    assert deploy.model_dir_for("1.2.3", None) == Path("models/1.2.3")
    assert deploy.model_dir_for("1.2.3", Path("x")) == Path("x")


def test_the_smoke_artifact_is_bakeable(project: Path) -> None:
    art = deploy.load_artifact(write_artifact(project, "artifacts/smoke/model"))
    assert (art.version, art.tag) == ("unreleased", "house-price-api:unreleased")


@pytest.mark.parametrize("relative", ["artifacts/smoke/candidates/ridge", "models/staging",
                                      "models/candidates/ridge", "elsewhere/model"])  # fmt: skip
def test_other_directories_are_refused(project: Path, relative: str) -> None:
    write_artifact(project, relative)
    with pytest.raises(DeployError, match="not a bakeable artifact directory"):
        deploy.load_artifact(Path(relative))


def test_an_absolute_path_is_refused(project: Path) -> None:
    path = write_artifact(project, "models/1.0.0")
    with pytest.raises(DeployError, match="not a bakeable"):
        deploy.load_artifact(project / path)


def test_a_candidate_role_is_refused(project: Path) -> None:
    from house_price.persistence import metadata

    links = metadata.MlflowLinks(refit="r1", final_holdout_evaluation=None)
    path = write_artifact(project, "models/9.9.9", artifact_role="candidate", holdout=None,
                          baseline_reference=None, temporal_diagnostic=None,
                          quality_gates=None, mlflow=links)  # fmt: skip
    with pytest.raises(DeployError, match="candidate artifact"):
        deploy.load_artifact(path)


def test_a_tampered_model_is_refused(project: Path) -> None:
    path = write_artifact(project, "artifacts/smoke/model")
    with (project / path / "model.joblib").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(DeployError, match="SHA-256"):
        deploy.load_artifact(path)


def test_a_version_mismatch_is_refused(project: Path) -> None:
    path = write_artifact(project, "artifacts/smoke/model")
    with pytest.raises(DeployError, match="MODEL_VERSION '1.0.0' != metadata.model_version"):
        deploy.load_artifact(path, "1.0.0")


# --------------------------------------------------------------------------- build


def test_build_checks_the_version_label(project: Path) -> None:
    art = deploy.load_artifact(write_artifact(project, "artifacts/smoke/model"))
    docker = FakeDocker({("docker", "image", "inspect", "--format", TEMPLATE): (0, "unreleased", ""),
                         ("docker", "image", "inspect", "--format", "{{.Config.User}}"): (0, "appuser", "")})  # fmt: skip
    assert deploy.build(art, docker) == "house-price-api:unreleased"
    (build,) = [c for c in docker.commands if c[:2] == ["docker", "build"]]
    assert "MODEL_DIR=artifacts/smoke/model" in build and "MODEL_VERSION=unreleased" in build
    assert not docker.ran("docker", "push") and not docker.ran("docker", "login")


def test_build_fails_on_a_wrong_label(project: Path) -> None:
    art = deploy.load_artifact(write_artifact(project, "artifacts/smoke/model"))
    docker = FakeDocker({("docker", "image", "inspect"): (0, "1.0.0", "")})
    with pytest.raises(DeployError, match="!= metadata.model_version"):
        deploy.build(art, docker)


def test_build_fails_on_a_root_user(project: Path) -> None:
    art = deploy.load_artifact(write_artifact(project, "artifacts/smoke/model"))
    docker = FakeDocker({("docker", "image", "inspect", "--format", TEMPLATE): (0, "unreleased", ""),
                         ("docker", "image", "inspect", "--format", "{{.Config.User}}"): (0, "", "")})  # fmt: skip
    with pytest.raises(DeployError, match="expected appuser"):
        deploy.build(art, docker)


def test_build_failure_is_reported(project: Path) -> None:
    art = deploy.load_artifact(write_artifact(project, "artifacts/smoke/model"))
    with pytest.raises(DeployError, match="docker build failed"):
        deploy.build(art, FakeDocker({("docker", "build"): (1, "", "boom")}))


# ---------------------------------------------------------------------------- push


def _push_docker(exists: tuple[int, str, str] = (1, "", "manifest unknown")) -> FakeDocker:
    return FakeDocker({("docker", "image", "inspect"): (0, "1.0.0", ""),
                       ("docker", "manifest", "inspect"): exists,
                       ("git", "remote", "get-url"): (0, "https://github.com/RNiyati07/house-price-prediction.git", "")})  # fmt: skip


def test_a_release_is_pushed_to_ghcr(project: Path) -> None:
    docker = _push_docker()
    reference = deploy.push(release_artifact(project), runner=docker, environ={})
    assert reference == "ghcr.io/rniyati07/house-price-api:1.0.0"
    assert docker.ran("docker", "manifest", "inspect", reference)  # checked before pushing
    assert docker.ran("docker", "push", reference)


def test_the_smoke_artifact_is_never_pushed(project: Path) -> None:
    art = deploy.load_artifact(write_artifact(project, "artifacts/smoke/model"))
    docker = _push_docker()
    with pytest.raises(DeployError) as caught:
        deploy.push(art, runner=docker, environ={})
    message = str(caught.value)
    assert "is_release is false" in message and "not a released x.y.z" in message
    assert not docker.ran("docker", "push") and not docker.ran("docker", "tag")


def test_a_non_release_versioned_artifact_is_refused(project: Path) -> None:
    path = write_artifact(project, "models/1.0.0", model_version="1.0.0", is_release=False)
    docker = _push_docker()
    with pytest.raises(DeployError, match="is_release is false"):
        deploy.push(deploy.load_artifact(path, "1.0.0"), runner=docker, environ={})
    assert not docker.ran("docker", "push")


def test_an_existing_tag_is_never_overwritten(project: Path) -> None:
    docker = _push_docker(exists=(0, '{"schemaVersion": 2}', ""))
    with pytest.raises(DeployError, match="already exists in GHCR"):
        deploy.push(release_artifact(project), runner=docker, environ={})
    assert not docker.ran("docker", "push")


def test_missing_authentication_stops_the_push(project: Path) -> None:
    docker = _push_docker(exists=(1, "", "unauthorized: authentication required"))
    with pytest.raises(DeployError, match="docker login ghcr.io"):
        deploy.push(release_artifact(project), runner=docker, environ={})
    assert not docker.ran("docker", "push")


def test_a_failed_push_explains_authentication(project: Path) -> None:
    docker = _push_docker()
    docker.answers[("docker", "push")] = (1, "", "denied: permission_denied")
    with pytest.raises(DeployError, match="docker login ghcr.io"):
        deploy.push(release_artifact(project), runner=docker, environ={})


@pytest.mark.parametrize("variable", ["CI", "GITHUB_ACTIONS"])
def test_ci_never_pushes(project: Path, variable: str) -> None:
    docker = _push_docker()
    with pytest.raises(DeployError, match="refusing to push from CI"):
        deploy.push(release_artifact(project), runner=docker, environ={variable: "true"})
    assert docker.commands == []


def test_a_local_image_with_another_label_is_refused(project: Path) -> None:
    docker = _push_docker()
    docker.answers[("docker", "image", "inspect")] = (0, "unreleased", "")
    with pytest.raises(DeployError, match="does not carry label 1.0.0"):
        deploy.push(release_artifact(project), runner=docker, environ={})


def test_ghcr_owner_from_the_remote() -> None:
    ssh = FakeDocker({("git",): (0, "git@github.com:Some-Owner/repo.git", "")})
    assert deploy.ghcr_owner(ssh) == "some-owner"
    with pytest.raises(DeployError, match="not a GitHub repository"):
        deploy.ghcr_owner(FakeDocker({("git",): (0, "https://example.com/x/y.git", "")}))


def test_cli_push_requires_a_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        deploy.main(["push"])
    assert "push requires --version" in capsys.readouterr().err


def test_cli_reports_errors(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert deploy.main(["build"]) == 1  # no smoke artifact here
    assert "metadata not found" in capsys.readouterr().err


# ------------------------------------------------------------------ the M12 files


def test_dockerfile_follows_doc04() -> None:
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS builder" in text and "AS runtime" in text
    assert text.count("ARG PYTHON_IMAGE=python:3.12-slim") == 1
    lock_copy = text.index("COPY pyproject.toml uv.lock ./")
    deps = text.index("RUN uv sync --frozen --no-dev --no-install-project")
    src = text.index("COPY src/ src/")
    assert lock_copy < deps < src  # dependency layer before the code (caching)
    for required in ("libgomp1", "--uid 10001", "USER appuser", "EXPOSE 8000", "HEALTHCHECK",
                     "org.opencontainers.image.version", "ARG MODEL_DIR", "ARG MODEL_VERSION",
                     "--workers 1", "--no-access-log", '"sh", "-c"', "${PORT:-8000}"):  # fmt: skip
        assert required in text, required
    model = text.index("COPY ${MODEL_DIR}/model.joblib")
    assert model > text.index("COPY configs/") and model > text.index("COPY --from=builder")
    assert "gunicorn" not in text.lower() and "curl" not in text.split("HEALTHCHECK")[1]
    assert "HPP_ALLOW_NON_RELEASE" not in text and "MODEL_VERSION=" not in text.split("LABEL")[1]
    for secret in ("TOKEN", "PASSWORD", "SECRET"):
        assert secret not in text.upper()


def test_dockerignore_admits_only_what_the_image_needs() -> None:
    rules = [line.strip() for line in (REPO_ROOT / ".dockerignore").read_text().splitlines()
             if line.strip() and not line.startswith("#")]  # fmt: skip
    assert rules[0] == "*"
    allowed = {r[1:] for r in rules if r.startswith("!")}
    assert allowed == {"pyproject.toml", "uv.lock", "src/", "configs/", "models/",
                       "artifacts/smoke/model/"}  # fmt: skip
    assert {"models/staging/", "models/candidates/"} <= set(rules)


def test_ci_docker_stage_never_pushes() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["docker"]
    assert job["needs"] == "smoke-train"
    assert job["permissions"] == {"contents": "read"} == workflow["permissions"]
    steps = " ".join(str(step) for step in job["steps"])
    assert "smoke-model" in steps and "deploy build" in steps and "deploy test" in steps
    raw = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in raw.splitlines()).lower()  # no comments
    for forbidden in ("docker push", "docker login", "deploy push", "packages: write",
                      "docker/login-action", "docker/build-push-action", "render"):  # fmt: skip
        assert forbidden not in code, forbidden


def test_runtime_dependencies_are_pinned_to_the_artifact_versions() -> None:
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pins = dict(d.split("==") for d in project["dependencies"])
    recorded = artifact.library_versions()
    for name in artifact.LIBRARIES:
        assert pins[name] == recorded[name], name
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    for name in artifact.LIBRARIES:
        assert f'name = "{name}"\nversion = "{pins[name]}"' in lock, name
