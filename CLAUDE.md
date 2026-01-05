# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository contains Claude Agent Skills for LLM engineering tasks. Skills are modular, filesystem-based capabilities that extend Claude's functionality with domain-specific expertise.

## Repository Files

- `REFERENCES.md` - External resources and documentation referenced across skills. **Update this file when creating new skills that reference external sources.**
- `.claude-plugin/marketplace.json` - Plugin manifest listing all available skills. **Update this file whenever a new skill is added.**

## Skill Structure

Each skill follows this structure:

```
skills/
└── skill-name/
    ├── SKILL.md           # Required: main instructions with YAML frontmatter
    ├── reference/         # Optional: detailed documentation
    └── scripts/           # Optional: utility scripts
```

### SKILL.md Requirements

```yaml
---
name: skill-name              # lowercase, hyphens only, max 64 chars
description: Brief description of what this Skill does and when to use it. Max 1024 chars.
---

[Markdown instructions - keep under 500 lines]
```

**Naming**: Use simple, descriptive names (e.g., `pytorch`, `transformers`, `lora`, `mlx`, `rlhf`)

**Description**: Write in third person. Include both what the skill does AND when Claude should use it.

### Recommended Sections

Skills should include these sections where applicable:

- **Table of Contents** - Required for files over 100 lines
- **Core Concepts** - Foundational knowledge for the topic
- **Practical Patterns** - Code examples and usage patterns
- **Best Practices** - Numbered list of key recommendations
- **References** - Links to files in `reference/` directory

## Authoring Guidelines

- Be concise - assume Claude already knows common concepts
- Use progressive disclosure: SKILL.md provides overview, reference files contain details
- Keep references one level deep from SKILL.md
- Include table of contents for files over 100 lines
- Always use forward slashes in paths (Unix-style)
- For executable scripts: make clear if Claude should run vs read them

### Degrees of Freedom

- **High freedom** (text instructions): When multiple approaches are valid
- **Medium freedom** (pseudocode/templates): When a preferred pattern exists
- **Low freedom** (specific scripts): When operations are fragile or consistency is critical
