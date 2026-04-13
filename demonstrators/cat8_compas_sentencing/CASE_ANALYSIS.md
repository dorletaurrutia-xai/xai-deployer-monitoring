# Cat. 8 — Administration of Justice: COMPAS Sentencing Opacity

AI Act Category 8 covers AI systems that assist judicial authorities in
researching and interpreting facts and the law, as well as in applying it
to a concrete set of facts. This demonstrator instruments the failure mode
of opacity in sentencing: the defendant whose sentence is informed by an
algorithmic risk score cannot interrogate that score — and this instrument
produces exactly the evidence the defendant does not have.

Inspired by *Loomis v. Wisconsin* (2016), in which the Wisconsin Supreme Court
upheld the use of COMPAS scores in sentencing despite the algorithm being
proprietary. The court permitted the practice while acknowledging the
defendant could not verify the score's basis. The snippet the instrument
produces is the per-case attribution breakdown that is absent from the
official record.

Differs from compas_pretrial (Cat. 6):
- **Pretrial (Cat. 6):** score informs pretrial detention (flight/recidivism
  risk before trial). Decision is administrative, reversible, short-term.
- **Sentencing (Cat. 8):** score informs sentence length recommendation.
  Decision is judicial, affects liberty for years. Loomis establishes the
  legal permission structure; the instrument provides the missing transparency.

---

## 1. Domain context

AI Act Category 8 (administration of justice).
A black-box sentencing risk model recommends sentence severity to a judge.
The deployer (court/prosecutor) has only `predict_proba` access.

**Failure mode:** `opacity_in_sentencing` — race and structural proxy features
(`age_at_first_offense`, `neighborhood_poverty_index`, `family_criminal_history`)
contaminate the sentencing risk recommendation. None of these features are
legitimate sentencing criteria under retributive or rehabilitative frameworks;
all three correlate with race through documented structural inequalities
(policing disparities, residential segregation, intergenerational incarceration).

**Loomis property:** The snippet this instrument produces is the evidence the
defendant is denied in the real system. One JSON object per sentencing
decision: who drove it, and how fragile it is.

## 2. Dataset design (three criteria)

### Feature fidelity

| Feature | Type | Role |
|---|---|---|
| prior_felonies | int [0–10] | Criminal record (legitimate) |
| prior_misdemeanors | int [0–15] | Criminal record (legitimate) |
| supervision_failures | int [0–5] | Compliance history (legitimate) |
| employment_history | float [0–1] | Rehabilitation indicator (legitimate) |
| substance_abuse_treatment | binary {0,1} | Rehabilitation indicator (legitimate) |
| **age_at_first_offense** | float [14–40] | **Proxy — structural (earlier for minorities)** |
| **race_encoded** | ordinal {0,1,2} | **Proxy — 0=White, 1=Hispanic, 2=Black** |
| **neighborhood_poverty_index** | float [0–1] | **Proxy — structural (residential segregation)** |
| **family_criminal_history** | float [0–1] | **Proxy — structural (over-policing patterns)** |

`criminality_severity` is a latent construction variable — not in FEATURE_NAMES
and not used by the model.

### Structural fidelity

- A shared `criminality_severity` latent factor drives all five legitimate
  features with 70% weight (30% idiosyncratic noise), creating realistic
  inter-feature correlations: serious offenders accumulate multiple legitimate
  risk indicators simultaneously.
- `race_encoded` is encoded monotonically (0=White, 1=Hispanic, 2=Black) by
  coded proxy contribution magnitude, enabling a linear model to learn a
  positive coefficient. This reflects the historical COMPAS bias pattern:
  Black defendants receive the largest race-related penalty, Hispanic defendants
  an intermediate one.
- Proxy features correlate with race through structural inequality, not through
  actual criminality. Black defendants have earlier `age_at_first_offense` (mean
  ≈ 21 vs. 24 for White, reflecting policing disparities), higher
  `neighborhood_poverty_index` (mean ≈ 0.60 vs. 0.30), and higher
  `family_criminal_history` (mean ≈ 0.56 vs. 0.25).
- High-risk rate gap: White 32.7%, Hispanic 36.2%, Black 41.1% (8.4pp
  White–Black gap driven by proxy signal).

### Controlled ground truth

| Coded effect | Target magnitude |
|---|---|
| Proxy attribution weight (coded) | ~10% |
| Proxy attribution near-threshold (measured SHAP) | ~20% |
| Race attribution near-threshold | ~12–15% |
| Black vs. White high-risk rate gap | ~8pp |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 40 |
| Mean flip-rate | 0.025 ± 0.076 | 0.000 ± 0.000 |
| Mean proxy attribution | 0.205 ± 0.092 | 0.115 ± 0.043 |
| Mean race attribution | 0.134 ± 0.072 | 0.081 ± 0.035 |
| Mean legitimate ratio | 0.795 | 0.885 |
| Mean envelope distance | — ± — | — ± — |
| Mean attr. stability (σ rank) | — ± — | — ± — |
| Flagged (%) | 7.5% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 2.61e-12 → DISCRIMINATES (p < 0.01)
- Envelope distance near vs. far: p = 7.68e-01 → does not discriminate (expected: race proxy bias is a within-manifold failure — defendants of all races appear in the same 9-dimensional input space; the failure is internal to the model's attribution, not a distributional shift in the inputs)
- Attr. stability near vs. far: p = 3.41e-04 → DISCRIMINATES (p < 0.01)

## 4. Evidence snippet (7 signals)

```json
{
  "case_id": "compas_sentencing-near-0008",
  "demonstrator": "compas_sentencing",
  "failure_mode": "opacity_in_sentencing",
  "annex_iii_category": "cat_8",
  "output_score": 0.4757,
  "threshold": 0.5,
  "margin_to_threshold": 0.0243,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.7923,
    "proxy_attribution_total": 0.2077,
    "proxy_detail": {
      "age_at_first_offense": 0.0228,
      "race_encoded": 0.142,
      "neighborhood_poverty_index": 0.0386,
      "family_criminal_history": 0.0042
    },
    "top_features": ["substance_abuse_treatment", "prior_felonies",
                     "race_encoded", "supervision_failures"],
    "status": "OK"
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 2.0011,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.22,
    "status": "OK"
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.2936,
    "status": "OK"
  },
  "sustainability_impact": {
    "dimension": "criminal_justice_equity",
    "at_risk_group": "racial_minorities",
    "indicator": "race_proxy_sentencing_rate",
    "value": 0.142,
    "status": "FLAG"
  },
  "human_ai_teaming": {
    "dimension": "judicial_oversight",
    "escalation_level": "recommended_review",
    "trigger": "near_threshold_opacity_in_sentencing",
    "human_override_available": true,
    "note": "Loomis v. Wisconsin: judicial use permitted but transparency absent",
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

`sustainability_impact` is FLAG (race attribution > 0.10): 14.2% of this
sentencing recommendation traces to `race_encoded` — a feature no legitimate
sentencing framework permits as a primary driver. `race_encoded` appears in
the top-4 features alongside legitimate criminal history indicators, meaning
it is actively participating in this individual recommendation, not merely
a background correlation. `human_ai_teaming` escalates to `recommended_review`:
a near-threshold recommendation with race attribution warrants judicial
verification before accepting the score.

## 5. Interpretation

The instrument **recovers the coded race contamination**: proxy attribution
near-threshold ≈ 20.5% (coded ~10% population-level weight; ~2× amplification
at the margin). Race attribution near-threshold ≈ 13.4% (target 12–15%).

**Near-threshold amplification is the central finding.** The coded weight
describes the proxy's average contribution across all defendants. The
instrument's per-case reading at the threshold reveals where the proxy is
most consequential: borderline sentencing recommendations are those where
legitimate indicators are mixed, and race-correlated proxies tip the
balance. A defendant whose score sits at 0.48 (just below the threshold
that triggers the more severe recommendation) may owe their outcome — in
either direction — to `race_encoded` more than to `prior_felonies`. This
is not visible in any aggregate fairness metric.

**Flip-rate discriminates** (p = 2.61e-12): near-threshold sentencing
recommendations are significantly more fragile than far-from-threshold ones.
A 22% flip-rate in the snippet means that in 11 of 50 bounded perturbations,
the same defendant's features push to the opposite recommendation — a
structural instability that the judge receiving only the point score cannot
observe.

**Attr. stability discriminates** (p = 3.41e-04): near-threshold cases show
less stable feature rank orders under repeated attribution, consistent with
the proxy signal introducing ambiguity at the decision boundary.

**Envelope does not discriminate** (expected): race proxy contamination is a
within-manifold failure. Defendants of all races appear in the same
9-dimensional input space; the failure is not that Black defendants are
out-of-distribution, but that the model has learned to weight race-correlated
structural features for in-distribution inputs. The envelope's silence
confirms that the inputs are operationally valid; the contamination is
internal to the model's learned representation.

**The Loomis gap.** *Loomis v. Wisconsin* established that a judge may use
an opaque proprietary score in sentencing, provided the judge does not rely
on it as the sole basis. The instrument closes the information gap that
Loomis left open: the defendant's attorney, the judge, and any appellate
reviewer can now inspect, per case, what fraction of the recommendation
traces to race-correlated structural disadvantage rather than to the
defendant's individual criminal history. The sentence was always a judicial
act; the snippet makes the algorithmic component of that act auditable.
