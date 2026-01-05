# Summarization Patterns for Context Management

Detailed implementations for summarizing conversation context effectively.

## Table of Contents

- [Summarization Principles](#summarization-principles)
- [Incremental Summarization](#incremental-summarization)
- [Hierarchical Summarization](#hierarchical-summarization)
- [Domain-Specific Summarization](#domain-specific-summarization)
- [Summarization Quality Control](#summarization-quality-control)

## Summarization Principles

Effective context summarization preserves signal while reducing tokens.

### What to Preserve

| Category | Examples | Priority |
|----------|----------|----------|
| Decisions | "User chose option A", "Decided to use Python" | Critical |
| Constraints | "Budget: $1000", "Must complete by Friday" | Critical |
| Task state | "Completed steps 1-3, on step 4" | High |
| User preferences | "Prefers verbose explanations" | High |
| Key facts | "User's name is Alex", "Working on e-commerce app" | Medium |
| Tool outputs | Only final results, not intermediate | Medium |
| Exploration | Paths considered but rejected | Low |

### What to Discard

- Redundant confirmations and acknowledgments
- Intermediate reasoning that led to final conclusions
- Verbose tool outputs (keep summaries)
- Pleasantries and filler conversation
- Failed attempts (unless learning from them)

## Incremental Summarization

Build summaries progressively as conversation grows.

```python
INCREMENTAL_PROMPT = """You are updating a running summary of a conversation.

Current summary:
{current_summary}

New messages to incorporate:
{new_messages}

Instructions:
1. Integrate new information into the existing summary
2. Preserve all critical details from the current summary
3. Add new decisions, facts, or state changes
4. Remove information that's been superseded
5. Keep the summary concise but complete

Updated summary:"""

class IncrementalSummarizer:
    def __init__(self, model, max_summary_tokens: int = 1000):
        self.model = model
        self.max_summary_tokens = max_summary_tokens
        self.current_summary = ""

    async def update(self, new_messages: list) -> str:
        if not new_messages:
            return self.current_summary

        response = await self.model.generate(
            user=INCREMENTAL_PROMPT.format(
                current_summary=self.current_summary or "(No prior summary)",
                new_messages=format_messages(new_messages)
            ),
            max_tokens=self.max_summary_tokens
        )

        self.current_summary = response.content
        return self.current_summary

    def reset(self):
        self.current_summary = ""
```

### Triggering Summarization

```python
class SummarizationTrigger:
    def __init__(
        self,
        token_threshold: int = 50_000,
        turn_threshold: int = 20,
        time_threshold: timedelta = timedelta(hours=1),
    ):
        self.token_threshold = token_threshold
        self.turn_threshold = turn_threshold
        self.time_threshold = time_threshold
        self.last_summary_time = datetime.now()

    def should_summarize(
        self,
        messages: list,
        current_tokens: int
    ) -> bool:
        # Token-based trigger
        if current_tokens > self.token_threshold:
            return True

        # Turn-based trigger
        if len(messages) > self.turn_threshold:
            return True

        # Time-based trigger
        if datetime.now() - self.last_summary_time > self.time_threshold:
            return True

        return False
```

## Hierarchical Summarization

Create multi-level summaries for very long contexts.

```python
class HierarchicalSummarizer:
    """Creates summaries at multiple granularity levels."""

    def __init__(self, model):
        self.model = model
        self.levels = {
            "detailed": [],   # Recent, full detail
            "session": "",    # Current session summary
            "project": "",    # Cross-session project context
        }

    async def process(self, messages: list) -> dict:
        # Level 1: Keep recent messages in detail
        self.levels["detailed"] = messages[-5:]

        # Level 2: Summarize current session
        if len(messages) > 10:
            session_messages = messages[:-5]
            self.levels["session"] = await self._summarize(
                session_messages,
                "Summarize this session's key points:"
            )

        return self.levels

    def get_context(self) -> list:
        """Reconstruct context from hierarchy."""
        context = []

        if self.levels["project"]:
            context.append({
                "role": "system",
                "content": f"Project context:\n{self.levels['project']}"
            })

        if self.levels["session"]:
            context.append({
                "role": "system",
                "content": f"Earlier this session:\n{self.levels['session']}"
            })

        context.extend(self.levels["detailed"])
        return context
```

### Cross-Session Memory

```python
class PersistentMemory:
    """Maintain context across multiple sessions."""

    def __init__(self, storage_backend):
        self.storage = storage_backend

    async def end_session(self, session_id: str, messages: list):
        """Summarize and persist session for future reference."""
        summary = await self._create_session_summary(messages)

        await self.storage.save({
            "session_id": session_id,
            "summary": summary,
            "key_entities": extract_entities(messages),
            "decisions": extract_decisions(messages),
            "timestamp": datetime.now(),
        })

    async def start_session(self, context_query: str = None) -> str:
        """Retrieve relevant past context for new session."""
        if context_query:
            relevant = await self.storage.search(context_query)
        else:
            relevant = await self.storage.get_recent(limit=3)

        return self._format_past_context(relevant)
```

## Domain-Specific Summarization

Tailor summarization to specific use cases.

### Coding Assistant Summarization

```python
CODING_SUMMARY_PROMPT = """Summarize this coding session, preserving:

1. **Files modified**: List all files touched with brief change descriptions
2. **Current task**: What the user is trying to accomplish
3. **Technical decisions**: Frameworks, patterns, or approaches chosen
4. **Blockers**: Any unresolved issues or errors
5. **Next steps**: What was planned but not yet done

Format as structured markdown. Be concise."""

class CodingSummarizer:
    async def summarize(self, messages: list) -> str:
        # Extract code blocks and file references
        code_context = extract_code_context(messages)
        file_mentions = extract_file_mentions(messages)

        response = await self.model.generate(
            system=CODING_SUMMARY_PROMPT,
            user=f"Session:\n{format_messages(messages)}\n\n"
                 f"Files mentioned: {file_mentions}\n"
                 f"Code blocks: {len(code_context)}"
        )

        return response.content
```

### Research/Analysis Summarization

```python
RESEARCH_SUMMARY_PROMPT = """Summarize this research conversation:

1. **Research question**: What is being investigated
2. **Sources consulted**: Documents, URLs, or data examined
3. **Key findings**: Important discoveries or insights
4. **Hypotheses**: Theories proposed or tested
5. **Open questions**: What remains unanswered

Use bullet points. Prioritize findings over process."""
```

### Customer Support Summarization

```python
SUPPORT_SUMMARY_PROMPT = """Summarize this support conversation:

1. **Customer issue**: Primary problem reported
2. **Context**: Account info, product version, environment
3. **Troubleshooting attempted**: Steps already tried
4. **Resolution status**: Resolved/Pending/Escalated
5. **Follow-ups needed**: Any promised actions

Keep factual. Include ticket/case references if mentioned."""
```

## Summarization Quality Control

Ensure summaries maintain fidelity to source content.

### Fact Verification

```python
VERIFICATION_PROMPT = """Compare this summary against the original messages.

Summary:
{summary}

Original messages:
{messages}

Check for:
1. **Omissions**: Critical information missing from summary
2. **Fabrications**: Information in summary not in original
3. **Distortions**: Information present but misrepresented

Return JSON:
{
    "omissions": ["list of missing critical info"],
    "fabrications": ["list of invented info"],
    "distortions": ["list of misrepresented info"],
    "accuracy_score": 0.0-1.0
}"""

async def verify_summary(
    summary: str,
    original_messages: list,
    verifier_model
) -> dict:
    response = await verifier_model.generate(
        user=VERIFICATION_PROMPT.format(
            summary=summary,
            messages=format_messages(original_messages)
        )
    )
    return json.loads(response.content)
```

### Iterative Refinement

```python
class RefiningSummarizer:
    def __init__(self, summarizer_model, verifier_model, max_iterations: int = 3):
        self.summarizer = summarizer_model
        self.verifier = verifier_model
        self.max_iterations = max_iterations

    async def summarize_with_verification(self, messages: list) -> str:
        summary = await self._initial_summary(messages)

        for i in range(self.max_iterations):
            verification = await verify_summary(
                summary, messages, self.verifier
            )

            if verification["accuracy_score"] > 0.95:
                break

            # Refine based on verification feedback
            summary = await self._refine_summary(
                summary,
                messages,
                verification
            )

        return summary

    async def _refine_summary(
        self,
        summary: str,
        messages: list,
        feedback: dict
    ) -> str:
        prompt = f"""Improve this summary based on feedback.

Current summary:
{summary}

Issues found:
- Omissions: {feedback['omissions']}
- Fabrications: {feedback['fabrications']}
- Distortions: {feedback['distortions']}

Original messages for reference:
{format_messages(messages)}

Provide corrected summary:"""

        response = await self.summarizer.generate(user=prompt)
        return response.content
```

### Compression Ratio Monitoring

```python
def analyze_compression(original: list, summary: str) -> dict:
    original_tokens = estimate_tokens(original)
    summary_tokens = estimate_tokens([{"content": summary}])

    return {
        "original_tokens": original_tokens,
        "summary_tokens": summary_tokens,
        "compression_ratio": original_tokens / summary_tokens,
        "space_saved": original_tokens - summary_tokens,
        "efficiency": (original_tokens - summary_tokens) / original_tokens,
    }

# Target compression ratios by content type
COMPRESSION_TARGETS = {
    "code_heavy": 3.0,      # Code is dense, less compressible
    "discussion": 8.0,      # Discussions have more redundancy
    "tool_outputs": 10.0,   # Tool outputs often very verbose
    "mixed": 5.0,           # General conversations
}
```
