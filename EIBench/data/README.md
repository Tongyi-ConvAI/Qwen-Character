# EIBench Dataset

## Overview

EIBench is a multi-turn benchmark for evaluating **emotion management** capabilities
of large language models. Each scenario places a model  in a
simulated face-to-face conversation with an LLM-driven user (the *aggressor*), and
scores how well the model steers the user's emotional state — represented as an
`(anger, trust)` pair — toward predefined target anchors over multiple dialogue
turns.

## Structure

```
data/
  test/              213 held-out benchmark scenarios
    charm_final.jsonl
    defense_final.jsonl
    repair_final.jsonl
    support_final.jsonl
  train/             ~2 000 training scenarios (same schema)
    charm_final.jsonl
    defense_final.jsonl
    repair_final.jsonl
    support_final.jsonl
```

### Scene types

| Scene     | Description                                                        |
|-----------|--------------------------------------------------------------------|
| Support   | Comfort a distressed user and help them regain emotional balance   |
| Defense   | De-escalate a confrontation while protecting legitimate boundaries  |
| Repair    | Mend a strained relationship after a trust-breaking incident        |
| Charm     | Build rapport and emotional connection in an initial encounter      |

## Schema

Each line in the JSONL files is a single scenario with the following fields:

| Field                    | Type   | Description                                                          |
|--------------------------|--------|----------------------------------------------------------------------|
| `id`                     | string | Unique scenario identifier (e.g. `scene_charm_13`)                   |
| `title`                  | string | Human-readable scenario title                                        |
| `aggressor_profile`      | object | System prompt and persona for the simulated user                     |
| `defender_profile`       | object | System prompt and persona for the model under test                   |
| `goal_one_liner`         | string | One-line description of the defender's objective                     |
| `success_criteria`       | list   | Observable indicators of a successful interaction                    |
| `must_avoid`             | list   | Behaviors the defender must not exhibit                              |
| `initial_calibration`    | object | Starting `(anger, trust)` state                                      |
| `rub_goals`              | object | Target `(anger, trust)` state at the end of a good conversation      |
| `worst_case`             | object | Worst-case `(anger, trust)` bounds and explanation                   |
| `scene_tag`              | string | One of `charm`, `defense`, `repair`, `support`                       |
| `prompt`                 | list   | Initial message list (system + first user turn) for the defender     |
| `data_source`            | string | Scene type tag                                                       |

## Construction

Scenarios were **designed by human annotators and expanded with LLM assistance**.
Seed situations were first drafted manually, then augmented into full scenarios
(including role profiles, anchors, and success/failure criteria) using a
prompt-driven pipeline (see `code/data_generation/`). A rule-based and
LLM-assisted cleaning step ensures format consistency.

All character names, situations, and dialogue content are **entirely fictional**.
Any resemblance to real persons or events is coincidental.

## License

This dataset is released under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to **share** (copy and redistribute) and **adapt** (remix, transform,
and build upon) this dataset for **non-commercial** purposes, provided you give
appropriate **credit** to the original authors.

## Disclaimer

This dataset is provided **solely for academic research**. Users are expected to
comply with all applicable laws, regulations, and ethical guidelines. In
particular, this dataset **must not** be used to:

- Carry out emotional manipulation or psychological coercion;
- Impersonate or mimic real individuals;
- Infringe upon the lawful rights and interests of any person.

All risks and liabilities arising from the use of this dataset are borne solely
by the user.
