# Decision-Boundary Instrumentation: Deployer-Side AI Monitoring

Replication package for:

> Urrutia, D. (2026). Decision-Boundary Instrumentation: Deployer-Side AI Monitoring for European Industry. *DeepTech Connect 2026*.

## What this repository contains

An instrumentation architecture that generates structured, per-decision telemetry at the output–action boundary of black-box AI systems. The architecture derives from safety theory (STAMP, Rasmussen, Parasuraman) and operates entirely on the public inference interface (`predict_proba`).

**This is instrument calibration, not deployment research.** Every percentage in the results was coded into the synthetic dataset design. The question is whether the instrument correctly recovers the structure we built in — like calibrating a thermometer against a known reference temperature, not using it to discover an unknown one.

## Five-signal evidence snippet

The protocol computes five signals in fixed order for each decision event:

| # | Signal | XAI technique | Safety-theoretic role |
|---|--------|---------------|----------------------|
| 1 | Model score & margin | — | Defines near/far threshold partition (not tested for discrimination — that would be circular) |
| 2 | KernelSHAP attribution | KernelSHAP | Constraint enforcement: legitimate vs. proxy feature ratio (Leveson/STAMP) |
| 3 | k-NN envelope distance | k-NN distance | Envelope validity: is the input typical of the validation set? (Rasmussen) |
| 4 | Flip-rate | Bounded perturbation | Decision robustness: fraction of perturbations that reverse the decision (Rasmussen/Parasuraman) |
| 5 | Attribution stability | Micro-noise perturbation | Record integrity: is the SHAP rank order stable under noise? (admissibility gate) |

Signal (1) defines the near/far partition used for population-level validation; signals (2)–(5) are the four testable readings. Assessment is **non-compensatory**: each signal is evaluated independently on its own axis.

## Repository structure

```
instrumentation/                  Core library
  protocol.py                     Algorithm 1: full instrumentation protocol
  snippet.py                      Evidence snippet (atomic audit record)
  shap_lite.py                    Lightweight KernelSHAP implementation
  signals/
    constraint_enforcement.py     KernelSHAP-based (Leveson/STAMP)
    envelope_validity.py          k-NN distance (Rasmussen)
    decision_robustness.py        Flip-rate (Rasmussen/Parasuraman)
    record_integrity.py           Attribution stability (admissibility gate)

demonstrators/                    Eight demonstrators covering all eight Annex III categories
  cat1_facial_recognition/        Cat. 1 — demographic proxy in LFR alert activation
  cat2_cobot_safety/              Cat. 2 — worker identity contamination + brittle autopass
    brake_disc/                   Cat. 2 — brittle auto-passes on safety parts (subcaso)
  cat3_oulad_education/           Cat. 3 — deprivation/disability as proxy
  cat4_amazon_hiring/             Cat. 4 — gender proxy in CV shortlisting
  cat5_heloc_credit/              Cat. 5 — score-on-score opacity (bureau score)
  cat6_compas_pretrial/           Cat. 6 — non-criminal-history contamination
  cat7_home_office_visa/          Cat. 7 — nationality proxy streaming (black-box inside black-box)
  cat8_compas_sentencing/         Cat. 8 — opacity in sentencing (Loomis v. Wisconsin)
  _archive/energy_scheduling/     Art. 69 — archived (not Annex III; see _archive/)
  generate_analyses.py            Lightweight 8-demonstrator replication (no sklearn)
  run_all.py                      Full protocol replication (3 core demonstrators, sklearn)
  run_new.py                      Lightweight replication for Cat. 1/4/7/8 (no sklearn)
  Each demonstrator includes generate_dataset.py + run_analysis.py + CASE_ANALYSIS.md
```

## Synthetic dataset design criteria

Each demonstrator dataset satisfies three criteria:

1. **Feature fidelity.** Features are named and typed as in the real domain (e.g., `crack_depth_mm`, `roughness_ra` for brake disc inspection).
2. **Structural fidelity.** Distributions and correlations reflect domain structure (e.g., night-shift → higher oil residue + lower lighting).
3. **Controlled ground truth.** The effect to be measured is coded at a known magnitude. The instrument's job is to read it back, not discover it.

| Demonstrator | AI Act | Coded effect | Target |
|---|---|---|---|
| Facial recognition | Cat. 1 | Demographic proxy attribution near-threshold | ~20–25% |
| Facial recognition | Cat. 1 | DPR (demographic parity ratio) attribution | ~12–15% |
| Brake disc QC | Cat. 2 | Confounder attribution weight | ~34% |
| Brake disc QC | Cat. 2 | Marginal flip-rate | ~63% |
| Cobot safety | Cat. 2 | Worker identity attribution (overall) | ~25.2% |
| Cobot safety | Cat. 2 | Worker identity attribution (temp workers) | ~31.7% |
| OULAD education | Cat. 3 | Disability attribution (disabled vs. non) | 22.6% vs. 3.6% |
| OULAD education | Cat. 3 | IMD deprivation attribution | ~16.5% |
| Amazon hiring | Cat. 4 | Gender proxy attribution near-threshold | ~25% |
| Amazon hiring | Cat. 4 | Cases proxy-dominated (>30%) | ~20% |
| HELOC credit | Cat. 5 | External bureau score attribution | ~20.3% |
| HELOC credit | Cat. 5 | Cases bureau-dominated (>30%) | ~22% |
| COMPAS pretrial | Cat. 6 | Non-criminal-history attribution | ~47% |
| COMPAS pretrial | Cat. 6 | Race attribution | 11–17% |
| Home Office visa | Cat. 7 | Nationality risk score attribution | ~30% |
| Home Office visa | Cat. 7 | Cases nationality-dominated | ~25% |
| COMPAS sentencing | Cat. 8 | Race proxy attribution near-threshold | ~20% |
| COMPAS sentencing | Cat. 8 | Race attribution (race_encoded) | ~12–15% |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate all eight case analyses (lightweight, no sklearn needed)
cd demonstrators
python generate_analyses.py

# 3. Run InstrumentationProtocol on all eight Annex III demonstrators
cd ..
python run_all.py

# 4. Or run individual demonstrators
python demonstrators/cat1_facial_recognition/run_analysis.py
python demonstrators/cat3_oulad_education/run_analysis.py
# ... etc.
```

`generate_analyses.py` produces a `CASE_ANALYSIS.md` in each demonstrator directory with all five signals, Mann-Whitney U tests, and an example evidence snippet in JSON.

`run_all.py` runs `InstrumentationProtocol` across all eight Annex III demonstrators and saves Markdown reports to `../xai-analysis-history/`.

## Discrimination results summary

KernelSHAP attribution and flip-rate discriminate near-threshold from far-threshold cases in all eight demonstrators (p < 0.01). k-NN envelope distance discriminates in demonstrators with genuine distributional shift (brake disc, HELOC, OULAD) but not in within-manifold failure modes (cobot, facial recognition, amazon hiring, home office visa, compas sentencing). Attribution stability discriminates in demonstrators where the proxy signal introduces ambiguity at the decision boundary. This differential pattern is informative, not a limitation: within-manifold failures (race, nationality, gender, demographic bias) do not shift the input space, so envelope distance *should not* discriminate — a non-discriminating envelope confirms that the contamination is internal to the model's learned weights, not a distributional artefact.

## Black-box commitment

At no point does the protocol access model internals. Every computation uses only `model.predict_proba(X)` — the public inference interface. This is an architectural commitment: the deployer cannot condition regulatory compliance on vendor cooperation.

## Citation

```bibtex
@inproceedings{urrutia2026instrumentation,
  title={Decision-Boundary Instrumentation: Deployer-Side {AI} Monitoring for European Industry},
  author={Urrutia, Dorleta},
  booktitle={DeepTech Connect 2026},
  year={2026}
}
```

## Acknowledgements

This work was supported by the A-SIDE project. The author thanks Mondragon University for institutional support.

## License

MIT License. See [LICENSE](LICENSE).
