# Python Algorithms Pack

`python_algorithms` is a starter pack for the `Dolci Instruct Python Algorithms` slice inside `allenai/Dolci-Instruct-SFT`.

The important point is that this slice is broader than "algorithms" in the narrow textbook sense. The rows we inspected are plain two-turn coding tasks with:

- one user prompt
- one assistant Python solution
- broad coding coverage under `source_dataset = "Dolci Instruct Python Algorithms"`

Representative prompts include:

- stdin/stdout programs
- function implementation tasks
- parsing and transformation problems
- competitive-programming style tasks
- prompts with explicit implementation bans or structural constraints

So this pack should reproduce that broader coding-instruction slice first, then add our own executable verification layer.

## Intended scope

Observable source shape to mirror:

- `id`
- `messages`
- `source_dataset`
- `domain`

Internal pack shape to add:

- prompt text
- canonical Python solution
- public examples when present
- hidden tests or trusted reference implementations
- optional static checks for constraints like banned built-ins

The verifier layer is ours. It does not come directly from the Dolci rows.

## Internal slices

The pack should split the monolithic source subset into practical internal slices:

1. `stdin_stdout_programs`
2. `function_implementation`
3. `constraint_heavy_tasks`
4. `parsing_transformations`
5. `competitive_problem_solving`

This is a better fit for the observed data than a too-narrow taxonomy like only graph algorithms or only dynamic programming.

## Start here

1. Mirror the plain two-turn coding format first.
2. Start with `stdin_stdout_programs` and `function_implementation`, since they give the cleanest execution contracts.
3. Add hidden tests before adding paraphrases or stylistic surface variation.
4. Add constraint checks only after the basic execution harness is stable.

## Dolci inspirations

- `allenai/correct-python-sft-187k-decontam-v2_tmp_ids`
- surfaced in `allenai/Dolci-Instruct-SFT` as `source_dataset = "Dolci Instruct Python Algorithms"`

The scaffold build still produces starter rows and a guide artifact only. It does not implement real code execution yet.
