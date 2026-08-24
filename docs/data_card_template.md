# Data Card: Gym Advice Preference Dataset

- **Dataset name**: Gym Advice Strength & Technique Preference Dataset (`gym_advice.jsonl`)
- **Source**: Preference Alignment Lab (`data/gym_advice.jsonl` & `data/sample_preferences.jsonl`)
- **License/permission**: Open Educational Lab License (Internal Research & Practice)
- **Schema**: `PreferenceExample` Pydantic model (`prompt: str`, `chosen: str`, `rejected: str`, `metadata: dict[str, Any]`)
- **Labeling rubric**: `technique_over_ego` (Prioritizes joint safety, proper exercise biomechanics, controlled movement, and injury prevention over ego lifting, momentum, and dangerous advice)
- **Known biases**: 
  - Domain specificity restricted to strength training, bodybuilding, and powerlifting.
  - Verbosity bias: Chosen responses average 2-3 detailed sentences detailing proper form, while rejected responses tend to be shorter declarative assertions promoting improper technique.
- **Safety/PII checks**: No personally identifiable information (PII) present; dataset includes explicit safety guardrails against dangerous physical exercise advice (e.g. holding breath for full set, extreme spinal arching under heavy loads).
- **Train/validation/test split method**: `split_by_prompt` (Prompt-grouped deterministic split using random seed 42 with 80% train / 20% validation ratio to prevent data leakage across splits).
