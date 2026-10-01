# Blind Spots of Qwen3-4B: Nepali Units, Currency, and Calendar Arithmetic

This project is my submission for the Fatima Fellowship technical challenge, *Blind Spots of Frontier Models*.

I evaluated `Qwen/Qwen3-4B` on everyday Nepali quantitative reasoning tasks involving local measurement systems, currency, and calendar arithmetic. The study investigates whether the model's answers change across English, Devanagari Nepali, and Romanized Nepali, and whether providing relevant factual context improves its performance.

The focus is a specific, practical question: **Can a small open-weight language model reliably reason about everyday quantities that depend on Nepali conventions and local knowledge?**

## Research Questions

1. How accurately does Qwen3-4B solve everyday quantitative questions involving Nepali units, currency, and calendar conventions?
2. Does answer accuracy differ between English, Devanagari Nepali, and Romanized Nepali versions of equivalent questions?
3. How much does providing relevant factual context change performance?
4. Which domains and question types account for the observed errors?

## Experimental Setup

| Component | Setup |
|---|---|
| Model | `Qwen/Qwen3-4B` |
| Evaluation inputs | 300 prompts |
| Base questions | 50 |
| Language forms | English, Devanagari Nepali, Romanized Nepali |
| Context conditions | With context and without context |
| Domains | Five everyday Nepali quantitative-reasoning domains |
| Execution environment | Kaggle notebook |
| Analysis | Accuracy, domain-level errors, paired comparisons, bootstrap confidence intervals |

The 300 prompts were constructed from 50 base questions, with each question represented in three language forms and evaluated under two context conditions.

The questions were supplied by the user or generated outside this repository before the evaluation run. The repository contains the evaluation and analysis workflow; it does not claim to implement the original question-generation process.

## Methodology

### 1. Controlled evaluation

The same underlying question is represented across the three language forms. This makes it possible to compare language conditions while keeping the intended task as consistent as possible.

Each language form is evaluated with and without relevant factual context. The context condition is intended to test whether supplying missing local facts helps distinguish knowledge limitations from reasoning difficulties.

### 2. Model execution

The notebook loads the specified Qwen3-4B model and executes the prepared evaluation prompts. Responses and associated evaluation metadata are saved for subsequent analysis.

### 3. Scoring and error analysis

The analysis examines overall answer accuracy, performance by domain and language, and the difference between context and no-context conditions.

Paired comparisons use corresponding variants of the same base question rather than treating every prompt as an independent underlying task. Bootstrap confidence intervals are used to communicate uncertainty in estimated performance differences.

Numerical answers involving ambiguous local-unit conventions require particular care. Questions whose expected answers depend on an unstated or disputed conversion convention should not be treated as unambiguous correctness tests.

## Findings and Interpretation

The evaluation is designed to identify concrete failure patterns in Nepali quantitative reasoning rather than assign a single general-purpose capability score to the model.

The main comparisons are:

- **Language variation:** Whether equivalent questions receive different accuracy across English, Devanagari Nepali, and Romanized Nepali.
- **Context availability:** Whether relevant factual context improves answer accuracy.
- **Domain variation:** Whether errors cluster around particular local units, conversions, currency calculations, or calendar arithmetic.
- **Error types:** Whether incorrect answers reflect missing facts, incorrect conversion procedures, arithmetic mistakes, or inconsistent interpretation of the question.

These findings should be interpreted within the scope of the evaluated question set. They do not establish that Qwen3-4B fails at all Nepali-language tasks, nor do they demonstrate a general safety failure.

## Repository Contents

The main artifacts produced by the evaluation workflow include:

- `README.md` — project overview, methodology, and interpretation.
- `results_qwen3_4b.jsonl` — individual evaluation outputs.
- `eval.json` — evaluation summary.
- `analysis/headline_accuracy.csv` — headline accuracy results.
- `analysis/accuracy_by_domain.csv` — accuracy broken down by domain.
- `analysis/paired_gaps.csv` — paired performance comparisons.
- `analysis/failure_taxonomy.csv` — categorized failure patterns.
- `analysis/manual_review_sample.csv` — sample selected for manual review.
- `run_manifest.json` — run configuration and reproducibility metadata.
- `qwen3_4b_artifacts.zip` — packaged evaluation artifacts.

The exact files available depend on the artifacts included in the published repository or release.

## Reproducibility

The repository is intended to make the model evaluation and analysis workflow inspectable and reproducible.

To reproduce the experiment:

1. Open the evaluation notebook in a compatible Kaggle environment.
2. Configure access to `Qwen/Qwen3-4B` and the required runtime dependencies.
3. Make the prepared evaluation inputs available to the notebook.
4. Run the model evaluation and scoring cells.
5. Run the analysis cells to regenerate the summary tables and error-analysis artifacts.

The notebook and run manifest should be consulted for the exact runtime settings, generation parameters, dependencies, and hardware constraints used in a particular run.

**Input provenance:** The questions were supplied by the user or generated outside this repository. Reproducing the evaluation therefore requires access to the same prepared inputs; the repository does not claim that those questions can be regenerated from its code alone.

## Limitations

- The evaluation covers a finite set of 50 base questions and their language/context variants.
- Three language forms do not represent every dialect, spelling variation, or code-mixed query a Nepali speaker might use.
- A small question set can reveal useful failure cases but cannot establish population-wide model performance.
- Results may depend on prompt wording, generation settings, and the exact factual context supplied.
- Some traditional measurement conversions can vary by convention or locality. Ambiguous cases must be excluded from headline accuracy or interpreted separately.
- Bootstrap intervals quantify uncertainty under the chosen resampling procedure; they do not eliminate limitations in question coverage or dataset construction.
- This is a single-model evaluation, not a comparison across all frontier models.

## Model and Artifacts

- Model: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Evaluation artifacts: [Fatima Fellowship Challenge — Hugging Face Bucket](https://huggingface.co/buckets/Prapti-T/Fatima-Fellowship-Challenge)

## Acknowledgment

This project was developed as a technical challenge submission exploring concrete capability gaps in language-model reasoning about everyday Nepali quantities.
