# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

A collection of **custom Claude Code agents and skills** — markdown-based configuration files that extend Claude Code's capabilities with domain-specific expertise. No application code, build system, or test suite.

## Agents vs Skills

**Agents** (subagents) change *who does the work* — separate AI workers with their own isolated context window, custom system prompt, specific tools, and independent permissions. They do not see the parent conversation history. Subagents cannot spawn other subagents.

**Skills** change what Claude *knows* — modular knowledge packages injected into Claude's existing conversation context. They run inline and have full access to conversation history.

Both are auto-invoked via the `description` field — no `@` or `/` prefix required.

## Structure

```
agents/     — Subagent definitions (markdown with YAML frontmatter)
skills/     — Skill definitions (SKILL.md + optional PATTERNS.md and references/)
```

## Agents (`agents/`)

Each `.md` file defines a specialized subagent. Install by copying to `~/.claude/agents/` (user-level) or `.claude/agents/` (project-level).

### Current agents

| Agent | Focus | Model |
|---|---|---|
| `frontend-developer` | Angular 21+ (standalone components, signals, OnPush) | sonnet |
| `typescript-pro` | Advanced TypeScript patterns for Angular 21+ | sonnet |
| `python-pro` | Python 3.11+ with FastAPI | sonnet |
| `fastapi-expert-agent` | Advanced FastAPI production architecture | opus |
| `nestjs-expert` | NestJS server-side applications | sonnet |
| `ui-ux-designer` | Research-backed UI/UX critique (read-only) | opus |
| `tailwind-patterns` | Tailwind CSS v4 utilities and patterns | inherit |
| `baml-expert-agent` | BAML (Boundary ML) applications | sonnet |
| `riskfolio-expert` | Riskfolio-Lib portfolio optimization | sonnet |
| `flyio-fastapi-deployment-expert` | Fly.io deployment for FastAPI | opus |
| `vercel-deployment-specialist` | Vercel deployment and edge functions | sonnet |

### Agent frontmatter fields

```yaml
name: kebab-case-identifier          # required
description: "Use proactively..."    # required — drives auto-invocation
tools: Read, Write, Edit, Bash, ...  # optional — inherits all if omitted
model: sonnet | opus | haiku | inherit  # optional — defaults to inherit
skills:                              # optional — injects skill content at startup
  - solid-principles
```

### Description convention

All agent descriptions follow this pattern for proactive auto-delegation:
```
"Use proactively whenever the user [trigger condition]. Do not wait to be asked;
delegate [domain] work to this agent automatically. Covers [capabilities]."
```

### Cross-agent architecture

- All coding agents preload the `solid-principles` skill via the `skills` frontmatter field (SOLID, TDD, clean code enforced at startup)
- `ui-ux-designer` is the only read-only agent (tools: Read, Grep, Glob) — no skill injection needed
- `python-pro` handles general Python + FastAPI; `fastapi-expert-agent` handles advanced production patterns with concrete code scaffolding
- `frontend-developer` and `typescript-pro` are both Angular 21+ specific — frontend-developer for components/templates, typescript-pro for advanced type system work
- Each coding agent has an "Integration with Other Agents" section listing only agents that exist in this repository

## Skills (`skills/`)

Each skill is a directory with `SKILL.md` as the entry point. Install by copying to `~/.claude/skills/` (user-level) or `.claude/skills/` (project-level).

### Current skills

| Skill | Focus | Auto-invoked |
|---|---|---|
| `solid-principles` | SOLID, TDD, clean code (universal) | Always — on every coding activity |
| `python-expert` | Python best practices with FastAPI | When writing Python |
| `skfolio` | skfolio portfolio optimization (scikit-learn API) | When using skfolio |
| `yfinance` | Yahoo Finance data retrieval | When using yfinance |

### Skill frontmatter fields

```yaml
name: kebab-case-identifier          # optional — defaults to directory name
description: "Load proactively..."   # recommended — drives auto-invocation
allowed-tools: Read, Write, Edit     # optional
```

### Supporting files

- `PATTERNS.md` — Concrete implementation patterns and code examples (used by `skfolio`, `yfinance`)
- `references/` — Deep-dive topic documents linked from SKILL.md (used by `solid-principles`)

## Conventions

- All files are pure markdown — no code to compile or test
- Agent filenames match the `name` field (e.g., `python-pro.md` → `name: python-pro`)
- Skills use `SKILL.md` as the entry point; directory name matches the skill name
- Descriptions must include proactive delegation language for auto-invocation
- No context-manager JSON protocols — agents discover context by exploring the codebase directly
- Agent integration sections reference only agents that exist in this repository
- `solid-principles` is preloaded into all coding agents via the `skills` frontmatter field
