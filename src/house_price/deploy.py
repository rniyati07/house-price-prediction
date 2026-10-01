"""Docker image build, container test and GHCR push (DOC-04 §11, §12, §16.5, §17; DOC-05 M12).

    python -m house_price.deploy build [--version x.y.z] [--model-dir DIR]   # make docker-build
    python -m house_price.deploy test  [--version x.y.z] [--model-dir DIR]   # make docker-test
    python -m house_price.deploy push  --version x.y.z                       # make docker-push

The model directory defaults to ``models/<version>`` when a version is given (a frozen
release, M13) and to the smoke artifact ``artifacts/smoke/model`` otherwise (CI and local
container tests only). ``build`` and ``test`` never touch a registry; ``push`` refuses
everything that is not a frozen, released, production artifact with a matching local image,
refuses a tag that already exists in GHCR, and refuses to run in CI.

Only the Docker CLI is driven from here; the service inside the image is the unchanged M11
application, which performs its own startup verification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from house_price.data.load import sha256_file
from house_price.persistence import artifact, metadata
from house_price.persistence.metadata import ArtifactMetadata

IMAGE = "house-price-api"
SMOKE_MODEL_DIR = Path("artifacts/smoke/model")
VERSION_LABEL = "org.opencontainers.image.version"
CONTAINER_PORT = 8000
RENDER_PORT = 10000  # DOC-05 M12: mimic Render's PORT handling
APP_UID = "10001"
HEALTHY_TIMEOUT_S = 120.0
HEALTH_FORMAT = "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}"

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class DeployError(RuntimeError):
    """A build, test or push check failed; the message says which and why."""


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True, check=False)


def _ok(runner: Runner, command: Sequence[str], what: str) -> str:
    result = runner(command)
    if result.returncode != 0:
        raise DeployError(f"{what} failed ({' '.join(command)}):\n{result.stderr.strip()}")
    return result.stdout.strip()


# ------------------------------------------------------------------------ artifact


@dataclass(frozen=True)
class Artifact:
    directory: Path
    metadata: ArtifactMetadata

    @property
    def version(self) -> str:
        return self.metadata.model_version

    @property
    def tag(self) -> str:
        return f"{IMAGE}:{self.version}"


def model_dir_for(version: str | None, model_dir: Path | None) -> Path:
    if model_dir is not None:
        return model_dir
    return artifact.MODELS_DIR / version if version else SMOKE_MODEL_DIR


def load_artifact(model_dir: Path, version: str | None = None) -> Artifact:
    """The artifact to bake in: a production artifact whose ``model.joblib`` matches its
    ``model_sha256`` and whose version equals ``version`` (when given). The directory must be
    one the build context admits (``.dockerignore``): ``models/<x.y.z>`` or the smoke
    artifact, as a relative path inside the repository."""
    posix = model_dir.as_posix().rstrip("/")
    allowed = posix == SMOKE_MODEL_DIR.as_posix() or re.fullmatch(r"models/[^/]+", posix)
    if model_dir.is_absolute() or not allowed or posix.startswith(("models/staging",
                                                                    "models/candidates")):  # fmt: skip
        raise DeployError(f"MODEL_DIR {posix!r} is not a bakeable artifact directory: use "
                          "models/<x.y.z> (a frozen release) or artifacts/smoke/model")  # fmt: skip
    meta = artifact.read_metadata(model_dir)
    if meta.artifact_role != "production":
        raise DeployError(f"{posix} holds a {meta.artifact_role} artifact; only the "
                          "production artifact of the selected model can be baked in")  # fmt: skip
    actual = sha256_file(model_dir / artifact.MODEL_FILE)
    if actual != meta.model_sha256:
        raise DeployError(f"{posix}/model.joblib SHA-256 {actual} != metadata {meta.model_sha256}")
    if version is not None and version != meta.model_version:
        raise DeployError(f"MODEL_VERSION {version!r} != metadata.model_version "
                          f"{meta.model_version!r} in {posix}")  # fmt: skip
    return Artifact(model_dir, meta)


# --------------------------------------------------------------------------- build


def build(art: Artifact, runner: Runner = run) -> str:
    """``docker build`` with MODEL_DIR / MODEL_VERSION, then check the OCI version label
    equals ``metadata.model_version`` (DOC-04 §11.2) and the image user is ``appuser``."""
    print(f"building {art.tag} from {art.directory.as_posix()} "
          f"(model_sha256 {art.metadata.model_sha256[:12]}, is_release={art.metadata.is_release})")  # fmt: skip
    command = ["docker", "build", "--build-arg", f"MODEL_DIR={art.directory.as_posix()}",
               "--build-arg", f"MODEL_VERSION={art.version}", "-t", art.tag, "."]  # fmt: skip
    # The real build streams its log (layer caching is visible there); tests inject a runner.
    result = subprocess.run(command, check=False) if runner is run else runner(command)
    if result.returncode != 0:
        raise DeployError(f"docker build failed: {' '.join(command)}")
    label = image_label(art.tag, runner)
    if label != art.version:
        raise DeployError(f"image label {VERSION_LABEL}={label!r} != metadata.model_version "
                          f"{art.version!r}")  # fmt: skip
    user = _ok(runner, ["docker", "image", "inspect", "--format", "{{.Config.User}}", art.tag],
               "image inspect")  # fmt: skip
    if user != "appuser":
        raise DeployError(f"image user is {user!r}, expected appuser (DOC-04 §11.4)")
    print(f"OK: {art.tag} built; {VERSION_LABEL}={label} equals metadata.model_version; "
          f"USER {user}")  # fmt: skip
    return art.tag


def image_label(tag: str, runner: Runner = run) -> str:
    template = '{{ index .Config.Labels "' + VERSION_LABEL + '" }}'
    return _ok(runner, ["docker", "image", "inspect", "--format", template, tag], "image inspect")


# ---------------------------------------------------------------------------- test


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _http(method: str, url: str, body: bytes | None = None) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=body, method=method,
                                     headers={"content-type": "application/json"})  # fmt: skip
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # localhost only
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def direct_prediction(model_dir: Path, config_dir: Path, example: dict[str, Any]) -> float:
    """The smoke pipeline's direct prediction of the example, outside any service
    (AC-060): the verified artifact's ``predict`` on the M11 frame of the example."""
    from house_price.api.predict import build_frame
    from house_price.config import SchemaConfig, load_model

    model, _ = artifact.load_verified(model_dir)
    schema = load_model(SchemaConfig, config_dir / "schema.yaml")
    return float(model.predict(build_frame([example], schema))[0])


IN_IMAGE_DIRECT = (
    "import json; from pathlib import Path; "
    "from house_price.persistence import artifact; "
    "from house_price.api.predict import build_frame; "
    "from house_price.config import SchemaConfig, load_model; "
    "m, _ = artifact.load_verified(Path('/app/model')); "
    "s = load_model(SchemaConfig, Path('/app/configs/schema.yaml')); "
    "e = json.loads(Path('/app/configs/api_example.json').read_text()); "
    "print(repr(float(m.predict(build_frame([e], s))[0])))"
)


@dataclass
class Container:
    """One running container of the image, removed on exit."""

    tag: str
    env: dict[str, str]
    container_port: int
    runner: Runner = run
    id: str = ""
    host_port: int = 0

    def __enter__(self) -> Self:
        self.host_port = _free_port()
        command = ["docker", "run", "-d", "-p", f"127.0.0.1:{self.host_port}:{self.container_port}"]
        for key, value in self.env.items():
            command += ["-e", f"{key}={value}"]
        self.id = _ok(self.runner, [*command, self.tag], "docker run")
        return self

    def __exit__(self, *exc: object) -> None:
        self.runner(["docker", "rm", "-f", self.id])

    def state(self) -> tuple[str, str]:
        out = _ok(self.runner, ["docker", "inspect", "--format", HEALTH_FORMAT, self.id], "docker inspect")  # fmt: skip
        status, _, health = out.partition(" ")
        return status, health

    def logs(self) -> str:
        result = self.runner(["docker", "logs", self.id])
        return result.stdout + result.stderr

    def wait_healthy(self, timeout: float = HEALTHY_TIMEOUT_S) -> float:
        """Until the Docker HEALTHCHECK (the image's own probe of ``/health`` on PORT) reports
        healthy; fails at once if the container exits."""
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            status, health = self.state()
            if status != "running":
                raise DeployError(f"container {status} before becoming healthy:\n{self.logs()}")
            if health == "healthy":
                return time.monotonic() - started
            time.sleep(1)
        raise DeployError(f"container not healthy within {timeout:.0f}s:\n{self.logs()}")

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.host_port}{path}"


def check_container(art: Artifact, config_dir: Path, *, port: int | None,
                    expected_price: float, runner: Runner = run) -> dict[str, Any]:  # fmt: skip
    """DOC-04 §16.5 steps 2-6 for one container (``port=None``: the image default)."""
    env = {"HPP_ALLOW_NON_RELEASE": "true"} if not art.metadata.is_release else {}
    if port is not None:
        env["PORT"] = str(port)
    container_port = port or CONTAINER_PORT
    example = json.loads((config_dir / "api_example.json").read_text(encoding="utf-8"))
    with Container(art.tag, env, container_port, runner) as container:
        seconds = container.wait_healthy()
        status, health = _http("GET", container.url("/health"))
        if status != 200 or health != {"status": "ok", "model_version": art.version}:
            raise DeployError(f"/health returned {status} {health}")
        uid = _ok(runner, ["docker", "exec", container.id, "id", "-u"], "docker exec id -u")
        if uid == "0" or uid != APP_UID:
            raise DeployError(f"server process runs as UID {uid}, expected {APP_UID} (AC-059)")
        ps = _ok(runner, ["docker", "exec", container.id, "sh", "-c",
                          "cat /proc/1/status | grep '^Uid:'"], "docker exec")  # fmt: skip
        if ps.split()[1] == "0":
            raise DeployError(f"PID 1 runs as root: {ps}")
        status, info = _http("GET", container.url("/model-info"))
        if status != 200 or info.get("model_sha256") != art.metadata.model_sha256:
            raise DeployError("the container does not serve the intended artifact "
                              f"(model_sha256 {info.get('model_sha256')})")  # fmt: skip
        status, result = _http("POST", container.url("/predict"), json.dumps(example).encode())
        if status != 200:
            raise DeployError(f"/predict returned {status}: {result}")
        if result["predicted_price"] != expected_price:  # exact (AC-060)
            raise DeployError(f"container price {result['predicted_price']!r} != direct "
                              f"prediction {expected_price!r}")  # fmt: skip
        logs = container.logs()
    return {"port": container_port, "healthy_after_s": round(seconds, 1), "uid": uid,
            "predicted_price": result["predicted_price"], "model_version": result["model_version"],
            "prediction_log_lines": logs.count('"event": "prediction.completed"')}  # fmt: skip


def check_startup_failure(art: Artifact, runner: Runner = run) -> str:
    """A wrong ``HPP_MODEL_DIR``: the container must exit non-zero with ``startup.failed``."""
    env = {"HPP_MODEL_DIR": "/app/no-such-model", "HPP_ALLOW_NON_RELEASE": "true"}
    with Container(art.tag, env, CONTAINER_PORT, runner) as container:
        deadline = time.monotonic() + HEALTHY_TIMEOUT_S
        while container.state()[0] == "running" and time.monotonic() < deadline:
            time.sleep(1)
        code = _ok(runner, ["docker", "inspect", "--format", "{{.State.ExitCode}}", container.id],
                   "docker inspect")  # fmt: skip
        logs = container.logs()
    if code == "0" or '"event": "startup.failed"' not in logs or "service.ready" in logs:
        raise DeployError(f"wrong HPP_MODEL_DIR did not fail startup (exit {code}):\n{logs}")
    step = re.search(r'"step": "([a-z_]+)"', logs)
    return f"exit code {code}, startup.failed at step {step.group(1) if step else '?'}"


def check_image_contents(tag: str, runner: Runner = run) -> str:
    """Only what serving needs: no data, tests, notebooks, git history or dev/training tools."""
    script = (
        "set -e; test \"$(ls -A /app | tr '\\n' ' ')\" = '.venv configs model src '; "
        "test \"$(ls -A /app/model | tr '\\n' ' ')\" = 'metadata.json model.joblib '; "
        "for p in /app/data /app/tests /app/notebooks /app/.git /app/mlruns /app/reports; do "
        "test ! -e $p; done; "
        "for m in mlflow optuna matplotlib pytest jupyter; do "
        '! python -c "import $m" 2>/dev/null; done; echo ok'
    )
    out = _ok(runner, ["docker", "run", "--rm", "--entrypoint", "sh", tag, "-c", script],
              "image contents check")  # fmt: skip
    if out.splitlines()[-1:] != ["ok"]:
        raise DeployError(f"unexpected image contents: {out}")
    return (
        "only .venv, configs, model, src; no data/tests/notebooks/.git; no dev or training packages"
    )


def test(art: Artifact, config_dir: Path = Path("configs"), runner: Runner = run) -> dict[str, Any]:
    """The container test (DOC-04 §16.5 steps 2-6; DOC-05 M12): the image default port and
    ``PORT=10000``; health, non-root UID, the served artifact, and the example price equal to
    the direct prediction both on this machine and inside the image; the startup failure on a
    wrong ``HPP_MODEL_DIR``; the image contents."""
    label = image_label(art.tag, runner)
    if label != art.version:
        raise DeployError(f"{art.tag} label {label!r} != metadata.model_version {art.version!r}")
    example = json.loads((config_dir / "api_example.json").read_text(encoding="utf-8"))
    host = direct_prediction(art.directory, config_dir, example)
    inside = float(_ok(runner, ["docker", "run", "--rm", "--entrypoint", "python", art.tag, "-c",
                                IN_IMAGE_DIRECT], "in-image direct prediction").splitlines()[-1])  # fmt: skip
    if inside != host:
        raise DeployError(f"direct prediction in the image {inside!r} != on this machine {host!r}")
    report: dict[str, Any] = {"image": art.tag, "label": label, "direct_prediction": host}
    report["default_port"] = check_container(art, config_dir, port=None, expected_price=host,
                                             runner=runner)  # fmt: skip
    report["render_port"] = check_container(art, config_dir, port=RENDER_PORT,
                                            expected_price=host, runner=runner)  # fmt: skip
    report["wrong_model_dir"] = check_startup_failure(art, runner)
    report["contents"] = check_image_contents(art.tag, runner)
    print(json.dumps(report, indent=2))
    print(f"OK: container test passed for {art.tag}")
    return report


# ---------------------------------------------------------------------------- push


def ghcr_owner(runner: Runner = run) -> str:
    """The GitHub owner of the ``origin`` remote (GHCR names are lower case)."""
    url = _ok(runner, ["git", "remote", "get-url", "origin"], "git remote get-url origin")
    match = re.search(r"github\.com[:/]([^/]+)/", url)
    if not match:
        raise DeployError(f"origin {url!r} is not a GitHub repository; pass --owner")
    return match.group(1).lower()


def remote_tag_exists(reference: str, runner: Runner = run) -> bool:
    """``docker manifest inspect``: found -> True; not found -> False; anything else (for
    example missing authentication) stops the push."""
    result = runner(["docker", "manifest", "inspect", reference])
    if result.returncode == 0:
        return True
    message = (result.stderr + result.stdout).lower()
    if "no such manifest" in message or "manifest unknown" in message or "not found" in message:
        return False
    raise DeployError(f"cannot check whether {reference} exists in GHCR:\n{message.strip()}\n"
                      "Authenticate first: echo $GHCR_TOKEN | docker login ghcr.io -u <user> "
                      "--password-stdin (a personal access token with write:packages)")  # fmt: skip


def push(art: Artifact, owner: str | None = None, runner: Runner = run,
         environ: dict[str, str] | None = None) -> str:  # fmt: skip
    """Push a release image to ``ghcr.io/<owner>/house-price-api:<version>`` (DOC-04 §12.3
    step 4). Refuses: CI; smoke, non-release, candidate or unreleased artifacts; a local image
    whose label differs; a tag that already exists in GHCR (tags are immutable)."""
    environ = dict(os.environ) if environ is None else environ
    if environ.get("CI") or environ.get("GITHUB_ACTIONS"):
        raise DeployError("refusing to push from CI: release images are pushed only from the "
                          "release machine (DOC-04 §12.4, M12-5)")  # fmt: skip
    meta = art.metadata
    problems = []
    if not meta.is_release:
        problems.append("is_release is false (smoke or staging artifact)")
    if not metadata.SEMVER.match(meta.model_version):
        problems.append(f"model_version {meta.model_version!r} is not a released x.y.z")
    if art.directory.as_posix().rstrip("/") != f"models/{meta.model_version}":
        problems.append(f"{art.directory.as_posix()} is not the frozen release directory "
                        f"models/{meta.model_version}")  # fmt: skip
    if meta.empty_fields():
        problems.append(f"metadata incomplete: {meta.empty_fields()}")
    if problems:
        raise DeployError("refusing to push:\n  - " + "\n  - ".join(problems))
    if image_label(art.tag, runner) != meta.model_version:
        raise DeployError(f"local image {art.tag} does not carry label {meta.model_version}; "
                          "run make docker-build and make docker-test first")  # fmt: skip
    reference = f"ghcr.io/{owner or ghcr_owner(runner)}/{IMAGE}:{meta.model_version}"
    if remote_tag_exists(reference, runner):
        raise DeployError(f"{reference} already exists in GHCR; tags are immutable, release a "
                          "new version instead (DOC-04 §17.3)")  # fmt: skip
    _ok(runner, ["docker", "tag", art.tag, reference], "docker tag")
    result = runner(["docker", "push", reference])
    if result.returncode != 0:
        raise DeployError(f"docker push {reference} failed:\n{result.stderr.strip()}\n"
                          "If this is an authentication error: docker login ghcr.io")  # fmt: skip
    print(f"pushed {reference}")
    return reference


# ----------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m house_price.deploy", description=__doc__)
    parser.add_argument("action", choices=["build", "test", "push"])
    parser.add_argument("--version", default=None, help="x.y.z; default: the smoke artifact")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="default: models/<version>, or artifacts/smoke/model")  # fmt: skip
    parser.add_argument("--owner", default=None, help="GHCR owner (default: origin remote)")
    args = parser.parse_args(argv)
    version = args.version or None
    if args.action == "push" and version is None:
        parser.error("push requires --version x.y.z")
    try:
        art = load_artifact(model_dir_for(version, args.model_dir), version)
        if args.action == "build":
            build(art)
        elif args.action == "test":
            test(art)
        else:
            push(art, args.owner)
    except (DeployError, artifact.ArtifactError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
