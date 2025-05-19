# Tasks tracker (edit per chunk)

| Date | Task | PR link | Status | Notes |
|------|------|---------|--------|-------|
| 2025-05-19 | Implement `/predict_fp` FastAPI route (#PRD-§3.1) | [PR #1](https://github.com/HarmoniqaOrg/VitronMax/pull/1) | ✅ Done | Initial implementation of BBB prediction endpoint |
| 2025-05-19 | Configure environment variables & secrets (#PRD-§3.6) | [PR #2](https://github.com/HarmoniqaOrg/VitronMax/pull/2) | ✅ Done (commit: e37a29d) | Added `.env.example`, `.env`, Pydantic config |
| 2025-05-19 | Set up CI/CD with GitHub Actions (#PRD-§3.6) | [PR #2](https://github.com/HarmoniqaOrg/VitronMax/pull/2) | ✅ Done (commit: e37a29d) | Tests, linting, Docker build (Note: CI implemented after first feature) |
| 2025-05-19 | Implement Supabase integration (#PRD-§3.6) | [PR #2](https://github.com/HarmoniqaOrg/VitronMax/pull/2) | ✅ Done (commit: e37a29d) | Added prediction storage client |
| 2025-05-19 | Configure Fly.io deployment (#PRD-§3.6) | [PR #2](https://github.com/HarmoniqaOrg/VitronMax/pull/2) | ✅ Done (commit: e37a29d) | Added `fly.toml` and documentation |
| 2025-05-19 | Implement `/batch_predict_csv` endpoint (#PRD-§3.2) | [PR #3](https://github.com/HarmoniqaOrg/VitronMax/pull/3) | ✅ Done (commit: f82c13e) | Async CSV batch processing with status tracking |
| 2025-05-19 | Test Fly.io deployment (#PRD-§3.6) | [PR #3](https://github.com/HarmoniqaOrg/VitronMax/pull/3) | ✅ Done (commit: 9c4d71a) | Fixed deployment dependencies, added python-multipart |
| 2025-05-19 | Deploy to production (#PRD-§3.6) | [PR #3](https://github.com/HarmoniqaOrg/VitronMax/pull/3) | ✅ Done | Successfully deployed to https://vitronmax.fly.dev/ |
| 2025-05-19 | Implement `/report` PDF generation endpoint (#PRD-§3.3) | [PR #4](https://github.com/HarmoniqaOrg/VitronMax/pull/4) | ✅ Done (commit: 3686405) | PDF report generation endpoint implemented |
| 2025-05-19 | Technical debt: Fix mypy and flake8 issues (Global Rules #1, #5) | [Future PR] | ⏳ Pending | Fix strict type checking, linting issues, and remove unused imports |
| 2025-05-19 | Technical debt: Make tests more resilient (Global Rules #5) | [PR #4](https://github.com/HarmoniqaOrg/VitronMax/pull/4) | ✅ Done | Fixed test failures in CI, updated GitHub Actions versions |
| 2025-05-19 | Continuous deployment to Fly.io (#PRD-§3.6) | [PR #4](https://github.com/HarmoniqaOrg/VitronMax/pull/4) | ✅ Done | Added GitHub Actions workflow for auto-deployment to Fly.io |
