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
| 2025-05-19 | Technical debt: Fix mypy and flake8 issues (Global Rules #1, #5) | [PR #5](https://github.com/HarmoniqaOrg/VitronMax/pull/5) | ✅ Done (commit: 6aef89c) | Replaced flake8 with ruff, fixed strict type checking, and added tests |
| 2025-05-19 | Technical debt: Make tests more resilient (Global Rules #5) | [PR #4](https://github.com/HarmoniqaOrg/VitronMax/pull/4) | ✅ Done | Fixed test failures in CI, updated GitHub Actions versions |
| 2025-05-19 | Continuous deployment to Fly.io (#PRD-§3.6) | [PR #4](https://github.com/HarmoniqaOrg/VitronMax/pull/4) | ✅ Done | Added GitHub Actions workflow for auto-deployment to Fly.io |
| 2025-05-20 | Add health check endpoint (#PRD-§3.6) | [PR #5](https://github.com/HarmoniqaOrg/VitronMax/pull/5) | ✅ Done (commit: 6aef89c) | Added `/healthz` endpoint for health checks |
| 2025-05-20 | Fix async handling of Supabase calls (#PRD-§3.1) | [PR #5](https://github.com/HarmoniqaOrg/VitronMax/pull/5) | ✅ Done (commit: 6aef89c) | Used asyncio.create_task for proper async handling |
| 2025-05-20 | Make CI/CD quality gates blocking (#PRD-§3.6, Global Rules #5) | [PR #5](https://github.com/HarmoniqaOrg/VitronMax/pull/5) | ✅ Done (commit: 6aef89c) | Removed || true from CI/CD pipeline, made tests and linting blocking |
| 2025-05-20 | Implement Supabase Storage for batch results (#PRD-§3.2) | [PR #6](https://github.com/HarmoniqaOrg/VitronMax/pull/6) | ✅ Done (commit: abcdef1) | Added storage_bucket_name to .env and implemented store_batch_result_csv method |
| 2025-05-20 | Update health check configuration in Fly.io (#PRD-§3.6) | [PR #6](https://github.com/HarmoniqaOrg/VitronMax/pull/6) | ✅ Done (commit: abcdef1) | Configured fly.toml to use /healthz endpoint for health checks |
| 2025-05-20 | Create SQL schema definition file (#PRD-§3.6) | [PR #6](https://github.com/HarmoniqaOrg/VitronMax/pull/6) | ✅ Done (commit: abcdef1) | Added db/schema.sql for database schema documentation |
| 2025-05-20 | Update documentation for new features (#PRD-§3.6, Global Rules #6) | [PR #6](https://github.com/HarmoniqaOrg/VitronMax/pull/6) | ✅ Done (commit: abcdef1) | Updated README.md and api-documentation.md with new functionality |
| 2025-05-20 | Fix deployment to fly at merge | [PR #7](https://github.com/HarmoniqaOrg/VitronMax/pull/7) | ✅ Done (commit: abcdef1) | Updated fly.toml to use /healthz endpoint for health checks |
| 2025-05-20 | Enable pytest-asyncio & un-skip async tests (#TechDebt-T1)        | [PR #11](https://github.com/HarmoniqaOrg/VitronMax/pull/11) | ✅ Done (commit: eb7039e) | Fixed async test, addressed pytest-asyncio deprecation |
| 2025-05-20 | Replace @app.on_event startup with lifespan handler (#TechDebt-T2)  | [PR #11](https://github.com/HarmoniqaOrg/VitronMax/pull/11) | ✅ Done (commit: eb7039e) | Migrated to lifespan events, silenced FastAPI deprecation warn |
| 2025-05-20 | Migrate Pydantic BaseSettings to ConfigDict (#TechDebt-T3)          | [PR #11](https://github.com/HarmoniqaOrg/VitronMax/pull/11) | ✅ Done (commit: eb7039e) | Migrated to ConfigDict, fixed Pydantic deprecation warn |
| 2025-05-20 | Convert fingerprint to NumPy array & add unit test (#PRD-§3.1, #TechDebt-T4) | [PR #12](https://github.com/HarmoniqaOrg/VitronMax/pull/12) | ✅ Done (commit: [hash_here]) | Fingerprint extraction now uses `np.ndarray[np.int8]`. |
| 2025-05-20 | Resolve Mypy errors & Harden CI (Global Rules #1, #5) | [PR #13](https://github.com/HarmoniqaOrg/VitronMax/pull/13) | ✅ Done (commit: dc1389f) | All Mypy errors fixed. CI: DeprecationWarnings as errors, action version pinned, code auto-formatted. |
| 2025-05-21 | Un-skip remaining batch tests (test_batch_predict_csv_valid, test_batch_status_valid) (#TechDebt-T1 P0) | [PR #17](https://github.com/HarmoniqaOrg/VitronMax/pull/17) | ✅ Done | Fixed underlying endpoint logic and test mocks. |
| 2025-05-21 | Fix batch CSV test failures and Supabase mocking issues (#PRD-§3.2) | [PR #XX](...) | ✅ Done (commit: [hash_here]) | Resolved all test failures in test_batch.py. |
| 2025-05-22 | Resolve final Mypy strict errors in integration tests (Global Rules #1, #5) | [PR #19](https://github.com/HarmoniqaOrg/VitronMax/pull/19) | ✅ Done (commit: 101b48c) | Fixed `no-untyped-def` for nested helper functions. |
| 2025-05-22 | Stabilize Fly.io deployment in CI (Global Rules #5) | [PR #19](https://github.com/HarmoniqaOrg/VitronMax/pull/19) | ✅ Done (commit: 101b48c) | Updated GitHub Action to `superfly/flyctl-actions/setup-flyctl@master`. |
| 2025-05-23 | Resolve all Mypy errors related to JobDetailsDict and BatchPredictionResponse typing (Global Rules #1) | [PR #20](https://github.com/HarmoniqaOrg/VitronMax/pull/20) | ✅ Done (commit: #88) | Ensured `progress` field in `JobDetailsDict` and correct type transformations. |
| 2025-05-23 | Unskip and pass batch processor status tests (#Test-C1 P2) | [PR #21](https://github.com/HarmoniqaOrg/VitronMax/pull/21) | ✅ Done (commit: #90) | Unskipped `test_get_job_status_not_found` and `test_get_job_status_found`. |
| 2025-05-20 | Secrets / env audit (#Infra-S1 P1) | [PR #XX](...) | ⏳ | Ensure STORAGE_BUCKET_NAME clear in .env.example & document required GitHub secrets. |
| 2025-05-20 | CI: Configure pytest to fail on skipped tests (#CI-S1 P1) | [PR #XX](...) | ⏳ | Make CI more strict regarding skipped tests. |
| 2025-05-20 | Docs refresh: Review and update all documentation (#Docs-R1 P1) | [PR #XX](...) | ⏳ | Ensure README, API docs, etc., are up-to-date. |
| 2025-05-20 | Increase test coverage to ≥ 80% (#Test-C1 P2) | [PR #XX](...) | ⏳ | Focus on db.py, batch.py. Current: ~60%. |
| 2025-05-20 | Implement CLI tool to purge old result CSVs from Supabase Storage (#Tool-T1 P2) | [PR #XX](...) | ⏳ | Cost control and bucket tidiness. |
| 2025-05-20 | Further tighten Mypy rules (#Mypy-S1 P3) | [PR #XX](...) | ⏳ | e.g., --warn-return-any, --disallow-any-generics. |
