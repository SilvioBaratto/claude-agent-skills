# Claude Code Agents & Skills

A curated collection of custom [Claude Code](https://claude.ai/code) agents and skills that extend Claude's capabilities with domain-specific expertise. Agents and skills are auto-invoked based on context — no manual commands required.

## Quick Start

### Install agents (Claude Code)

Copy agent files to your user-level or project-level directory:

```bash
# User-level (available in all projects)
cp agents/*.md ~/.claude/agents/

# Project-level (available in this project only)
mkdir -p .claude/agents
cp agents/*.md .claude/agents/
```

### Install agents (GitHub Copilot CLI)

This repo includes Copilot CLI custom agents under `.github/agents/*.agent.md`.
To install them **globally** (available in all folders on your machine):

```bash
python install_copilot_agents.py
```

This copies the agent profiles into `~/.copilot/agents/`.

### Install skills (Claude Code)

Copy skill directories to your user-level or project-level directory:

```bash
# User-level
cp -r skills/* ~/.claude/skills/

# Project-level
mkdir -p .claude/skills
cp -r skills/* .claude/skills/
```

Restart Claude Code or run `/agents` and `/skills` to verify installation.

### Install skills (GitHub Copilot CLI)

This repo includes Copilot CLI skills under `.github/skills/*`.
To install them **globally** (available in all folders on your machine):

```bash
python install_copilot_skills.py
```

This copies the skills into `~/.copilot/skills/`.

## Agents

Agents are specialized AI workers that run in their own isolated context window. Claude automatically delegates tasks to the right agent based on the `description` field — no `@` prefix needed.

### Frontend & UI

| Agent                  | Model   | Description                                                                                                                                      |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **frontend-developer** | Sonnet  | Angular 21+ development — standalone components, signals, `input()`/`output()`, OnPush change detection, native control flow, lazy-loaded routes |
| **typescript-pro**     | Sonnet  | Advanced TypeScript for Angular 21+ — signal typing, typed reactive forms, generic patterns, strict mode, monorepo architecture                  |
| **tailwind-patterns**  | Inherit | Tailwind CSS v4 — CSS-first `@theme` configuration, container queries, OKLCH colors, responsive design, dark mode                                |
| **ui-ux-designer**     | Opus    | Research-backed UI/UX critique — Nielsen Norman Group studies, accessibility audits, distinctive design direction (read-only)                    |

### Backend & APIs

| Agent                    | Model  | Description                                                                                                                      |
| ------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **python-pro**           | Sonnet | Python 3.11+ with FastAPI — async patterns, Pydantic v2, SQLAlchemy 2.0+, type safety, pytest                                    |
| **fastapi-expert-agent** | Opus   | Advanced FastAPI production architecture — application factory, repository pattern, JWT auth, WebSocket, Docker containerization |
| **nestjs-expert**        | Sonnet | NestJS server-side applications — API development, authentication, database integration, guards, interceptors                    |
| **baml-expert-agent**    | Sonnet | BAML (Boundary ML) — type-safe LLM functions, structured outputs, RAG pipelines, streaming, FastAPI integration                  |

### Deployment

| Agent                               | Model  | Description                                                                                        |
| ----------------------------------- | ------ | -------------------------------------------------------------------------------------------------- |
| **flyio-fastapi-deployment-expert** | Opus   | Fly.io deployment for FastAPI — containerization, Supabase integration, scaling, CI/CD, monitoring |
| **vercel-deployment-specialist**    | Sonnet | Vercel platform — edge functions, middleware, deployment strategies, performance optimization      |

### Supabase

| Agent                          | Model  | Description                                                                                                                                                                   |
| ------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **supabase-connection-expert** | Sonnet | Supabase Postgres connection setup — Session Pooler for SQLAlchemy/psycopg2, Transaction Pooler for Prisma/pgbouncer, NestJS integration, compute tier sizing for 1000+ users |
| **supabase-auth-linker**       | Haiku  | Supabase auth & security doc navigator — matches questions to the correct official documentation link (read-only)                                                             |

### Finance

| Agent                | Model  | Description                                                                                                 |
| -------------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| **riskfolio-expert** | Sonnet | Riskfolio-Lib portfolio optimization — mean-variance, hierarchical clustering, Black-Litterman, risk parity |

## Skills

Skills are knowledge packages injected into Claude's conversation context. They teach Claude domain-specific patterns and conventions. Auto-invoked when relevant.

| Skill                | Description                                                                                                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **solid-principles** | SOLID, TDD, clean code, code smells, design patterns. Preloaded into every coding agent via `skills` frontmatter. Includes `references/` with deep-dive docs on TDD, architecture, testing, and more |
| **python-expert**    | Modern Python best practices — type safety, clean architecture, security patterns, Pythonic idioms for production-grade development                                                                  |
| **skfolio**          | skfolio portfolio optimization using the scikit-learn API — optimization models, prior estimators, covariance estimation, model selection. Includes `PATTERNS.md` with code examples                 |
| **yfinance**         | Yahoo Finance data retrieval — tickers, price history, financial statements, screeners, sector data, caching. Includes `PATTERNS.md` with code examples                                              |

## How Auto-Invocation Works

Both agents and skills use the `description` field for automatic delegation. When your prompt matches an agent or skill description, Claude invokes it without any manual command.

**Agents** are delegated tasks and run in isolation:

```
You: "Build an Angular component for user profiles"
     → Claude automatically delegates to frontend-developer agent
```

**Skills** inject knowledge into the current conversation:

```
You: "Optimize this portfolio allocation"
     → Claude automatically loads skfolio skill into context
```

The `solid-principles` skill is special — it's preloaded into every coding agent at startup via the `skills` frontmatter field, so SOLID/TDD/clean code principles are enforced in every agent's context window.

## Project Structure

```
agents/
  frontend-developer.md        Angular 21+ components and templates
  typescript-pro.md             Advanced TypeScript for Angular
  python-pro.md                 Python + FastAPI development
  fastapi-expert-agent.md       FastAPI production architecture
  nestjs-expert.md              NestJS applications
  ui-ux-designer.md             UI/UX design critique (read-only)
  tailwind-patterns.md          Tailwind CSS v4 patterns
  baml-expert-agent.md          BAML/LLM applications
  riskfolio-expert.md           Portfolio optimization
  flyio-fastapi-deployment-expert.md   Fly.io deployment
  vercel-deployment-specialist.md      Vercel deployment
  supabase-connection-expert.md  Supabase connection config (FastAPI/NestJS/Prisma)
  supabase-auth-linker.md        Supabase auth & security doc links (read-only)
skills/
  solid-principles/
    SKILL.md                    Core principles (injected into all coding agents)
    references/                 Deep-dive docs (TDD, architecture, testing, etc.)
  python-expert/
    SKILL.md                    Python best practices
  skfolio/
    SKILL.md                    skfolio guidance
    PATTERNS.md                 Code examples
  yfinance/
    SKILL.md                    yfinance guidance
    PATTERNS.md                 Code examples
```

## Customization

### Create a new agent

```bash
mkdir -p ~/.claude/agents
cat > ~/.claude/agents/my-agent.md << 'EOF'
---
name: my-agent
description: "Use proactively whenever the user [trigger]. Do not wait to be asked; delegate [domain] work to this agent automatically."
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
skills:
  - solid-principles
---

Your system prompt here...
EOF
```

### Create a new skill

```bash
mkdir -p ~/.claude/skills/my-skill
cat > ~/.claude/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: "Load proactively whenever the user [trigger]. Do not wait to be asked; apply this skill automatically."
---

Your skill instructions here...
EOF
```

### Key frontmatter fields

**Agents:**

| Field         | Required | Description                                        |
| ------------- | -------- | -------------------------------------------------- |
| `name`        | Yes      | Kebab-case identifier                              |
| `description` | Yes      | Drives auto-invocation — include "Use proactively" |
| `tools`       | No       | Available tools (inherits all if omitted)          |
| `model`       | No       | `sonnet`, `opus`, `haiku`, or `inherit`            |
| `skills`      | No       | Skills injected at startup                         |

**Skills:**

| Field                      | Required    | Description                                         |
| -------------------------- | ----------- | --------------------------------------------------- |
| `name`                     | No          | Defaults to directory name                          |
| `description`              | Recommended | Drives auto-invocation — include "Load proactively" |
| `allowed-tools`            | No          | Tools available when skill is active                |
| `disable-model-invocation` | No          | Set `true` to prevent auto-loading                  |

## License

[MIT](LICENSE)
