# Deployment Procedure (GHCR and Render)

**Status:** prepared in M12; first used in the M13 Release Run. Source of truth: DOC-04 §11 (Docker image), §12 (Render deployment), §16.5 (container test), §17 (release process); DOC-05 M12 and M13.

Nothing in this document has been executed against a registry or Render yet. M12 builds and tests the image with the **smoke artifact** only, locally and in CI; a smoke image is never pushed and never deployed. The first push and the first Render deploy use the real, frozen artifact `models/x.y.z/` in M13.

---

## 1. Architecture

```
Developer machine                    GitHub                               Render
─────────────────                    ──────                               ──────
make train / evaluate / freeze
        │
        ▼
make docker-build VERSION=x.y.z  (MODEL_DIR=models/x.y.z, MODEL_VERSION=x.y.z)
        │  make docker-test VERSION=x.y.z   (local container test)
        ▼
make docker-push VERSION=x.y.z ──► ghcr.io/<owner>/house-price-api:x.y.z
git tag vx.y.z; git push --tags ─► repository (tag)
                                                       │
                                           Render Web Service (Docker image runtime)
                                           image: ghcr.io/<owner>/house-price-api:x.y.z
                                           health check path: /health
                                           instance: free tier
                                           env: HPP_LOG_LEVEL=INFO (PORT set by Render)
                                                       │
HTTPS client ──► https://<service>.onrender.com ──► container :$PORT
```

For this repository `<owner>` is `rniyati07` (the GitHub owner of `origin`; `make docker-push` derives it from the remote).

### Why the model artifact is baked into the image

- ADR-15 bakes the artifact into the image: there is no volume mount and no download at runtime (DOC-04 §11.3). The image contains `/app/model/{model.joblib, metadata.json}`, copied from `MODEL_DIR` as the last build layer.
- An image tag therefore identifies exactly one model: `house-price-api:1.0.0` contains `models/1.0.0` (NFR-012), and the OCI label `org.opencontainers.image.version` equals `metadata.model_version` (`make docker-build` checks this).
- At startup the service verifies the baked artifact exactly as M11 does (release flag, SHA-256, library versions, schema hash, warm-up prediction); a broken image never becomes healthy.

### Why Render does not build the image from the repository

- `models/` is gitignored (ADR-17): the frozen artifact is **outside git**. Render's build-from-repository mode would build an image **without** the model.
- So the image is built where the frozen artifact exists (the release machine) and Render is given the finished image (SD-06, SD-07; DOC-04 §12.2).
- The image is built once, tested with `make docker-test`, pushed, and deployed unchanged: **the tested image is the deployed image**, so there is no drift between what was tested and what runs (NFR-026).
- Render performs no build; it only pulls images (DOC-04 §12.4).

---

## 2. GitHub Container Registry (GHCR)

| Item | Value |
|---|---|
| Image | `ghcr.io/<owner>/house-price-api:x.y.z` (here `ghcr.io/rniyati07/house-price-api:x.y.z`) |
| Tag | The model version `x.y.z` = `metadata.model_version` = git tag `vx.y.z` without the `v` (DOC-04 §17.3) |
| Visibility | Public package: the image contains only public data derivatives, code and a model, and a public package needs no registry credentials in Render (DOC-04 §12.2) |
| Who pushes | Only the developer, from the release machine, with `make docker-push VERSION=x.y.z` |

Rules:

- Release images are pushed to GHCR; **CI never pushes** (the CI `docker` job has read-only permissions, no `docker login` and no push step; M12-5).
- Tags are **immutable by convention**: an existing tag is never overwritten. `make docker-push` checks the registry first (`docker manifest inspect`) and refuses a tag that already exists. A fix is released as a new version.
- `make docker-push` also refuses: a smoke, staging or other non-release artifact (`is_release` false); a version that is not `x.y.z`; an artifact directory other than `models/x.y.z`; candidate artifacts; a local image whose version label differs; and any run inside CI.

### Access (one-time, before M13)

1. Create a GitHub personal access token (classic) with the `write:packages` scope (it implies `read:packages`). Keep it out of the repository, the Dockerfile and build arguments.
2. Log in from the release machine:

   ```sh
   echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
   ```

3. After the first push, set the `house-price-api` package to **public** in the GitHub package settings.

If credentials are missing, `make docker-push` stops with an authentication message before anything is pushed.

---

## 3. Render Web Service settings (DOC-04 §12.6)

| Setting | Where | Value |
|---|---|---|
| Service type | Render dashboard | **Web Service**, runtime **Docker image** (deploy an existing image) |
| Image URL | Render service settings | `ghcr.io/<owner>/house-price-api:x.y.z` (the exact release tag; never `latest`) |
| Health check path | Render service settings | `/health` |
| Auto-deploy | Render service settings | **Off**: a deploy happens only when the tag is deliberately changed |
| Instance type | Render service settings | Free tier |
| `HPP_LOG_LEVEL` | Render environment | `INFO` |
| `PORT` | Set by Render | Render's value; the container listens on it (`--port ${PORT:-8000}`) |
| Everything else | Baked into the image | — |

Notes:

- No other environment variables are set. In particular there is **no model-version variable**: the version comes only from `metadata.json`, so it cannot disagree with the artifact (DOC-04 §11.6). `HPP_MODEL_DIR` and `HPP_CONFIG_DIR` keep their image defaults (`/app/model`, `/app/configs`), and `HPP_ALLOW_NON_RELEASE` is never set on Render (it exists only for the CI smoke container).
- The container runs **one Uvicorn worker** (SD-09) as the non-root user `appuser` (UID 10001), with Uvicorn's access log off; the service writes its own JSON logs to stdout, which Render collects.
- If startup verification fails, the container exits and the health check never passes; Render marks the deploy failed and keeps the previous deploy serving (DOC-04 §12.5). Confirm this behavior on the chosen instance type during the first deployment and record it in the release notes.
- The free tier stops idle services; the first request after idling includes a cold start with the full startup verification (accepted by ADR-15).

---

## 4. Release procedure (M13)

Run on the release machine after `make freeze VERSION=x.y.z` has produced the frozen artifact (DOC-04 §12.3, §17.2):

1. **Frozen artifact exists** at `models/x.y.z/` (`model.joblib`, `metadata.json` with `is_release: true`, `model_version: "x.y.z"`); quality gates QG-10 to QG-16 passed.
2. **Build the image** with `MODEL_DIR=models/x.y.z` and `MODEL_VERSION=x.y.z`:

   ```sh
   make docker-build VERSION=x.y.z
   ```

   The build fails unless the OCI version label equals `metadata.model_version`.
3. **Test the image** (release artifact, no override flag):

   ```sh
   make docker-test VERSION=x.y.z
   ```

   Checks: `/health` 200 on the default port and with `PORT=10000`; process UID ≠ 0 (AC-059); the served `model_sha256`; the example prediction equals the direct pipeline prediction (AC-060); a wrong `HPP_MODEL_DIR` fails startup; the image holds only `.venv`, `configs`, `model`, `src`.
4. **Push the image**:

   ```sh
   make docker-push VERSION=x.y.z
   ```

   Refused if the tag already exists in GHCR.
5. **Update Render** to that exact image tag: in the service settings set the image URL to `ghcr.io/<owner>/house-price-api:x.y.z`.
6. **Deploy** (manual deploy, or the service's deploy hook). Render pulls the image, starts the container and polls `/health`; the deploy goes live only after `/health` returns 200.
7. **Verify `/health`** over HTTPS:

   ```sh
   curl -s https://<service>.onrender.com/health
   ```

8. **Verify `/model-info`**:

   ```sh
   curl -s https://<service>.onrender.com/model-info
   ```

9. **Confirm the model version** in both responses equals `x.y.z` (AC-061).
10. **Record the deployment** in the release notes: image reference and digest, Render deploy time, the `/health` and `/model-info` results, and the startup-failure behavior observed on the instance type.

Tagging (`git tag vx.y.z && git push --tags`, tag commit = `metadata.git_commit`) and the documentation updates follow DOC-04 §17.2 steps 7 and 13.

---

## 5. Rollback (DOC-04 §12.6, §17.4)

1. In the Render service settings, set the image URL back to the **previous known-good tag**, `ghcr.io/<owner>/house-price-api:<previous x.y.z>`.
2. Deploy and verify `/health` and `/model-info` (the version must equal the previous version).
3. Record the rollback in the release notes.

Because each version is a complete, immutable image in GHCR, rollback involves no retraining and no rebuild. Do not rebuild an old commit to roll back.

---

## 6. Local and CI container test with the smoke artifact (M12)

```sh
make docker-build        # bakes artifacts/smoke/model (model_version "unreleased")
make docker-test         # runs it with HPP_ALLOW_NON_RELEASE=true, also with PORT=10000
```

The CI `docker` job does the same with the smoke artifact produced by the `smoke-train` job (`smoke-model`). The smoke image is local or CI-only: it is never pushed, never deployed, and never presented as a release.
