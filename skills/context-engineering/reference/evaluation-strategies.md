# Evaluation Strategies for Context Management

Testing context management effectiveness requires specific evaluation approaches beyond standard LLM benchmarks.

## Table of Contents

- [Baseline and Delta Comparisons](#baseline-and-delta-comparisons)
- [LLM-as-Judge Assessment](#llm-as-judge-assessment)
- [Transcript Replay Testing](#transcript-replay-testing)
- [Error Regression Tracking](#error-regression-tracking)
- [Token Pressure Checks](#token-pressure-checks)

## Baseline and Delta Comparisons

Compare agent performance with different context management strategies.

```python
class ContextEvaluator:
    def __init__(self, agent_factory, test_cases):
        self.agent_factory = agent_factory
        self.test_cases = test_cases

    async def compare_strategies(
        self,
        strategies: list[ContextStrategy]
    ) -> dict:
        results = {}

        for strategy in strategies:
            agent = self.agent_factory(context_strategy=strategy)
            metrics = await self._evaluate_agent(agent)
            results[strategy.name] = metrics

        return self._compute_deltas(results)

    async def _evaluate_agent(self, agent) -> dict:
        return {
            "task_completion_rate": await self._measure_completion(agent),
            "context_retention_accuracy": await self._measure_retention(agent),
            "average_tokens_per_task": await self._measure_token_usage(agent),
            "latency_p50": await self._measure_latency(agent),
        }
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Task completion rate | % of tasks completed successfully | >95% |
| Context retention accuracy | Ability to recall earlier information | >90% |
| Token efficiency | Tokens used per successful task | Minimize |
| Latency impact | Added time from context management | <10% overhead |

## LLM-as-Judge Assessment

Use an LLM to evaluate conversation quality and context coherence.

```python
JUDGE_PROMPT = """Evaluate this agent conversation for context management quality.

Rate each dimension 1-5:

1. **Coherence**: Does the agent maintain logical consistency?
2. **Memory**: Does it remember relevant earlier context?
3. **Relevance**: Are responses grounded in provided context?
4. **Efficiency**: Does it avoid redundant information requests?
5. **Recovery**: Does it handle context gaps gracefully?

Conversation:
{conversation}

Provide ratings and brief justifications for each."""

async def judge_conversation(conversation: list, judge_model) -> dict:
    response = await judge_model.generate(
        user=JUDGE_PROMPT.format(
            conversation=format_conversation(conversation)
        )
    )
    return parse_ratings(response.content)
```

### Automated Regression Detection

```python
class QualityGate:
    def __init__(self, baseline_scores: dict, threshold: float = 0.1):
        self.baseline = baseline_scores
        self.threshold = threshold

    def check(self, new_scores: dict) -> bool:
        """Return True if quality maintained, False if regression detected."""
        for metric, baseline_value in self.baseline.items():
            new_value = new_scores.get(metric, 0)
            delta = (baseline_value - new_value) / baseline_value

            if delta > self.threshold:
                print(f"Regression in {metric}: {baseline_value} -> {new_value}")
                return False

        return True
```

## Transcript Replay Testing

Replay historical conversations to test context management changes.

```python
class TranscriptReplayer:
    def __init__(self, agent, transcripts: list):
        self.agent = agent
        self.transcripts = transcripts

    async def replay_all(self) -> list[ReplayResult]:
        results = []

        for transcript in self.transcripts:
            result = await self._replay_single(transcript)
            results.append(result)

        return results

    async def _replay_single(self, transcript: dict) -> ReplayResult:
        """Replay a transcript and compare outputs."""
        original_outputs = transcript["assistant_messages"]
        new_outputs = []

        messages = []
        for turn in transcript["turns"]:
            messages.append(turn["user"])
            response = await self.agent.run(messages)
            new_outputs.append(response)
            messages.append(response)

        return ReplayResult(
            transcript_id=transcript["id"],
            original_outputs=original_outputs,
            new_outputs=new_outputs,
            similarity_scores=self._compute_similarity(
                original_outputs, new_outputs
            ),
        )
```

### Key Checkpoints

1. **Information preservation**: Does summarization retain critical facts?
2. **Decision consistency**: Would the agent make the same decisions?
3. **Tool usage patterns**: Are the same tools invoked appropriately?
4. **Error handling**: Are edge cases handled consistently?

## Error Regression Tracking

Track specific failure patterns related to context issues.

```python
class ContextErrorTracker:
    ERROR_TYPES = [
        "amnesia",           # Forgot earlier context
        "hallucination",     # Invented non-existent context
        "contradiction",     # Contradicted earlier statements
        "context_overflow",  # Hit token limits
        "retrieval_failure", # Failed to find stored context
    ]

    def __init__(self):
        self.errors = defaultdict(list)

    def log_error(
        self,
        error_type: str,
        conversation_id: str,
        turn_number: int,
        details: str
    ):
        self.errors[error_type].append({
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "details": details,
            "timestamp": datetime.now(),
        })

    def get_report(self) -> dict:
        return {
            error_type: {
                "count": len(errors),
                "examples": errors[:5],
            }
            for error_type, errors in self.errors.items()
        }
```

### Root Cause Analysis

```python
def analyze_context_failure(conversation: list, failure_turn: int) -> dict:
    """Analyze why context management failed at a specific turn."""

    # Extract context state at failure point
    context_at_failure = conversation[:failure_turn]

    return {
        "token_count": estimate_tokens(context_at_failure),
        "summary_count": count_summaries(context_at_failure),
        "information_density": compute_density(context_at_failure),
        "critical_info_present": check_critical_info(context_at_failure),
        "potential_causes": identify_causes(context_at_failure),
    }
```

## Token Pressure Checks

Test agent behavior as context limits are approached.

```python
class TokenPressureTest:
    def __init__(self, agent, context_limit: int):
        self.agent = agent
        self.context_limit = context_limit

    async def test_at_thresholds(self) -> dict:
        """Test agent behavior at various context fill levels."""
        thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        results = {}

        for threshold in thresholds:
            target_tokens = int(self.context_limit * threshold)
            result = await self._test_at_level(target_tokens)
            results[f"{int(threshold * 100)}%"] = result

        return results

    async def _test_at_level(self, target_tokens: int) -> dict:
        # Build context to target size
        messages = await self._build_context(target_tokens)

        # Test critical operations
        return {
            "retrieval_accuracy": await self._test_retrieval(messages),
            "new_info_integration": await self._test_integration(messages),
            "task_completion": await self._test_task(messages),
            "graceful_degradation": await self._test_degradation(messages),
        }
```

### Stress Test Scenarios

| Scenario | Description | Pass Criteria |
|----------|-------------|---------------|
| Long conversation | 50+ turn conversation | Maintains key facts |
| Information overload | Many tool results | Filters to relevant |
| Conflicting context | Contradictory information | Uses most recent/authoritative |
| Near-limit operation | 95%+ context used | Compacts without data loss |
| Recovery after compaction | Post-summary task | Completes successfully |

## Continuous Monitoring

```python
class ContextHealthMonitor:
    def __init__(self, metrics_backend):
        self.metrics = metrics_backend

    def record_turn(
        self,
        conversation_id: str,
        turn_number: int,
        token_count: int,
        was_compacted: bool,
        task_succeeded: bool
    ):
        self.metrics.record({
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "token_count": token_count,
            "compaction_event": was_compacted,
            "task_success": task_succeeded,
            "timestamp": datetime.now(),
        })

    def get_health_summary(self, time_window: timedelta) -> dict:
        recent = self.metrics.query(since=datetime.now() - time_window)

        return {
            "avg_tokens_per_conversation": mean(r["token_count"] for r in recent),
            "compaction_frequency": sum(r["compaction_event"] for r in recent) / len(recent),
            "success_rate": sum(r["task_success"] for r in recent) / len(recent),
            "token_limit_approaches": sum(1 for r in recent if r["token_count"] > LIMIT * 0.9),
        }
```
