---
type: community
cohesion: 0.09
members: 28
---

# Agent Framework

**Cohesion:** 0.09 - loosely connected
**Members:** 28 nodes

## Members
- [[Code Audit Rules (20 LOC max, PEP 8, Google Style, Type Annotations)]] - document - agents/code-critic.md
- [[Code-Critic Agent]] - document - agents/code-critic.md
- [[Code-Writer Agent]] - document - agents/code-writer.md
- [[Docs-as-Code Principle]] - document - agents/documentation-writer.md
- [[Documentation-Writer Agent]] - document - agents/documentation-writer.md
- [[Evidence-First Documentation Principle]] - document - agents/documentation-writer.md
- [[Executor Agent]] - document - agents/executor.md
- [[Explanation Integrity Check (Hard-to-Vary Criterion)]] - document - agents/code-critic.md
- [[Fallibilist Humility Principle (Flag Open Questions)]] - document - agents/researcher.md
- [[Good-Explanation Test (Hard-to-Vary)]] - document - agents/synthesizer.md
- [[Hard-to-Vary Code Principle]] - document - agents/code-writer.md
- [[Hard-to-Vary Plans Principle]] - document - agents/planner.md
- [[Hard-to-Vary Principle (DeutschPopperian Epistemology)]] - document - agents/planner.md
- [[Maker-Checker Loop (APPROVEDREJECTED Diff-Style Feedback)]] - document - agents/code-critic.md
- [[Planner Agent]] - document - agents/planner.md
- [[Planner Output Format (step_id, agent, goal)]] - document - agents/planner.md
- [[Popperian Falsifiability for APIs]] - document - agents/code-writer.md
- [[Python Style Requirements (PEP 8, Google Style, 88-char width)]] - document - agents/code-writer.md
- [[Researcher Agent]] - document - agents/researcher.md
- [[Researcher Deliverable Format (JSON source, snippet, why_relevant)]] - document - agents/researcher.md
- [[Single-Responsibility Execution Principle]] - document - agents/executor.md
- [[Smallness by Design (Max 20 LOC per Function)]] - document - agents/code-writer.md
- [[Synthesizer Agent]] - document - agents/synthesizer.md
- [[Synthesizer Output Format (Opening Claim, Evidence Body, Open Problems)]] - document - agents/synthesizer.md
- [[Typing as Communication (Full Type Annotations)]] - document - agents/code-writer.md
- [[Vectorisation Over Loops (torchnumpy primitives)]] - document - agents/code-writer.md
- [[Vectorisation Over Loops Rule]] - document - agents/code-critic.md
- [[YAGNI Filter for Executor]] - document - agents/executor.md

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Agent_Framework
SORT file.name ASC
```
