from typing import Any

from sdg.commons.starter_pack import (
    StarterPackSpec,
    StarterRowSpec,
    build_starter_pack,
    publish_starter_pack,
    summarize_starter_pack,
    verify_starter_pack,
)

SPEC = StarterPackSpec(
    name="python_algorithms",
    family="python_algorithms",
    description="Starter scaffold for Dolci-style Python coding tasks with executable verification.",
    goal=(
        "Reproduce the broad Dolci Instruct Python Algorithms slice: plain two-turn Python coding "
        "tasks spanning stdin/stdout programs, function implementations, parsing transforms, and "
        "constraint-heavy exercises, then add executable verification with hidden tests or trusted "
        "reference implementations."
    ),
    getting_started=(
        "Mirror the observable Dolci row shape first: one user prompt and one assistant Python solution.",
        "Split the slice by prompt contract, not by an imagined narrow topic taxonomy.",
        "Treat hidden tests as our verifier layer, not as something copied from the source subset.",
    ),
    starter_rows=(
        StarterRowSpec(
            title="STDIN and STDOUT programs",
            prompt=(
                "Start with tasks that read structured input from standard input and print the "
                "final answer to standard output. This matches a large visible portion of the "
                "Dolci Python slice and gives the verifier a stable execution contract."
            ),
            target=(
                "The first row shape should preserve the raw coding prompt, a canonical Python "
                "program, public I/O examples, and a hidden execution bundle that feeds input text "
                "to the program and compares exact stdout."
            ),
            primary_subset="stdin_stdout_programs",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Execute the solution as a program and compare stdout on hidden inputs.",
        ),
        StarterRowSpec(
            title="Function implementation tasks",
            prompt=(
                "Add prompts that ask for a function with a named signature and described behavior, "
                "rather than a full stdin/stdout script. This covers the other major observable form "
                "inside the Dolci coding subset."
            ),
            target=(
                "Keep prompt text, canonical implementation, public examples, and a hidden import-and-call "
                "test bundle together so the verifier can exercise the requested function directly."
            ),
            primary_subset="function_implementation",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Import the generated function and run hidden behavioral tests against it.",
        ),
        StarterRowSpec(
            title="Constraint-heavy coding prompts",
            prompt=(
                "Reserve a slice for prompts with explicit implementation constraints such as "
                "forbidden built-ins, custom sorting, manual parsing, or case-normalization rules. "
                "These appear frequently in the source subset and matter for prompt difficulty."
            ),
            target=(
                "The verifier should check both correctness and any declared implementation bans or "
                "required structure, so the row contract needs room for lint-like constraint checks "
                "in addition to standard hidden tests."
            ),
            primary_subset="constraint_heavy_tasks",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Run hidden tests and static constraint checks against the submitted code.",
        ),
        StarterRowSpec(
            title="Parsing and transformation tasks",
            prompt=(
                "Carve out prompts that primarily transform nested JSON, strings, tables, logs, or "
                "other structured inputs into a required output format. This is broader than "
                "algorithms in the narrow sense, but it is clearly part of the Dolci coding slice."
            ),
            target=(
                "Use a row contract that can distinguish parsing contract, output formatting rules, "
                "and hidden adversarial inputs, since these tasks often fail on edge formatting rather "
                "than algorithmic complexity alone."
            ),
            primary_subset="parsing_transformations",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Execute hidden transformation cases and compare exact structured outputs.",
        ),
        StarterRowSpec(
            title="Competitive problem solving",
            prompt=(
                "Keep a separate slice for classic greedy, search, dynamic programming, and numeric "
                "problem-solving prompts from competitive-programming style sources. This is still "
                "important, but it should be only one slice inside the broader coding pack."
            ),
            target=(
                "The pack should support canonical solver code plus hidden cases that stress edge "
                "conditions, performance-sensitive branches, and exact output formatting."
            ),
            primary_subset="competitive_problem_solving",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Run hidden judge cases and compare outputs to the trusted solver.",
        ),
    ),
)


def build(cfg: dict[str, Any]):
    return build_starter_pack(cfg, spec=SPEC)


def verify(run_id_or_path: str):
    return verify_starter_pack(run_id_or_path, spec=SPEC)


def summarize(run_id_or_path: str):
    return summarize_starter_pack(run_id_or_path)


def publish(run_id_or_path: str, out_dir: str | None = None):
    return publish_starter_pack(run_id_or_path, out_dir=out_dir)
