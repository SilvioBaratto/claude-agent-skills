# Copilot instructions for this repository

## Useful commands
- Regenerate/refresh repo instructions:
  - `copilot init` (interactive equivalent: `/init`)
  - Inspect which instruction files are active: `/instructions`
- Install Copilot CLI agents globally (user-level):
  - `python install_copilot_agents.py`
  - Dry run: `python install_copilot_agents.py --dry-run`
- Install Copilot CLI skills globally (user-level):
  - `python install_copilot_skills.py`
  - Dry run: `python install_copilot_skills.py --dry-run`

## High-level architecture
- This repository is a library of **agents** and **skills**, maintained in two parallel formats:
  - Claude Code sources:
    - `agents\*.md`
    - `skills\<skill>\SKILL.md` (plus optional `PATTERNS.md`, `references\`)
  - Copilot CLI equivalents (used by the installer scripts):
    - `.github\agents\*.agent.md`
    - `.github\skills\<skill>\SKILL.md` (plus optional supporting files)
- The Python installer scripts only copy from `.github\agents\` and `.github\skills\` into the user-level Copilot directories:
  - `~/.copilot/agents/`
  - `~/.copilot/skills/`

## Key conventions (repo-specific)
- Agent naming:
  - Agent filename matches YAML `name` (kebab-case).
  - `description` is written to trigger auto-selection (starts with “Use proactively…”).
- Skill shape:
  - Each skill is a directory; entry point is always `SKILL.md` with YAML frontmatter (`name`, `description`).
  - Skill descriptions use “Load proactively…”.
- Keeping Claude vs Copilot copies aligned:
  - If you change behavior/content of an agent/skill, update both the Claude Code source and the Copilot CLI copy.
  - Copilot agent tool names are lowercase (`read`, `edit`, `execute`, `search`, `web`) and differ from Claude’s tool labels.
- Tool restrictions are intentional:
  - `ui-ux-designer` and `supabase-auth-linker` are meant to remain read-only; keep limited tools in both variants.
