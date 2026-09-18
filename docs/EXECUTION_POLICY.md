# Execution Policy

## 1. Primary Model

The locally fine-tuned QLoRA model is the primary inference model.

The system must attempt local inference before invoking any fallback provider.

## 2. Gemini Fallback

Gemini is not the default model.

Gemini may be invoked only when the local model cannot provide a sufficiently grounded answer according to the application's fallback policy.

## 3. No Fabricated Answers

The system must not invent financial facts, numerical values, formulas, citations, or source information.

If sufficient evidence is unavailable, the system must explicitly indicate that the answer cannot be established from the available information.

## 4. Retrieval

When retrieval is enabled, retrieved evidence must be available to the generation layer before an answer is produced.

The model must not claim that information came from a source unless that source was actually retrieved.

## 5. Numerical Reasoning

Financial numerical answers must preserve the original numerical values, units, and calculation context.

When execution-based reasoning is enabled, generated calculations must be validated before their result is presented.

## 6. Fallback Hierarchy

The intended inference hierarchy is:

1. Local fine-tuned model
2. Retrieval / grounding layer
3. Validated execution layer for numerical reasoning
4. Gemini fallback when configured and permitted
5. Explicit uncertainty response

The exact routing policy will be implemented and benchmarked separately.

## 7. Evaluation Isolation

Evaluation datasets and test questions must not be used as training data unless explicitly designated as training data.

Test-set contamination must be avoided.

## 8. Reproducibility

Experiments must record:

- model identifier
- adapter identifier
- dataset version
- configuration
- random seed
- software environment
- evaluation dataset
- evaluation metrics

## 9. Secrets

API keys, tokens, credentials, and private configuration must never be committed to Git.

## 10. Production Claims

No performance claim may be made without a reproducible evaluation result and clearly identified baseline.