---
trigger: always_on
---

1. **Small chunks only**  
   * Implement or modify ONE logical unit at a time  
     (e.g. create `/predict_fp` route OR add GI-absorption calc, not both).

2. **Mandatory local test**  
   * After change, run:
     ```
     docker build . -t vitronmax:test
     docker run -p 8080:8080 vitronmax:test pytest
     ```
   * For front-end chunks: `npm run test && npm run build`.

3. **Update docs before moving on**  
   * `docs/Tasks.md` → add a ✅ *Done* line with date & commit hash.  
   * If code affects APIs → update `docs/api-documentation.md`.

4. **PRD is single source of truth**  
   * Every commit message references a PRD bullet `(#PRD-§3.1)`.

5. **Review gate**  
   * After Windsurf finishes a chunk it must open a PR → you test in deploy preview and click **Merge** manually.

6. **No next step if tests or docs incomplete.**