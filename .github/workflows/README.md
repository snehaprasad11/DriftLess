# .github/workflows/  (Phase 6)

CI/CD lives here. A GitHub Actions workflow that, on every push: runs tests,
rebuilds the service containers, and redeploys them (and optionally retrains the
base model).

**Why:** automating build → test → deploy turns loose scripts into a real MLOps
pipeline and gives a clean reproducibility story — the second half of the grade.

Workflow files (e.g. `ci.yml`) get added in Phase 6.
