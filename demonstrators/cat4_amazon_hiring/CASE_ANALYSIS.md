# Cat. 4 — Employment: Amazon Hiring Gender Proxy Bias

AI Act Category 4 covers AI systems used in employment and workers management, including
automated CV screening and candidate shortlisting. This demonstrator instruments a
gender-proxy contamination failure: the model is trained on historical selection data
in which male-coded linguistic and social signals correlate with acceptance, and
learns to exploit those signals as shortcuts.

Inspired by Amazon's documented ML hiring tool (2014–2018), which learned to penalise
CVs containing signals correlated with female gender because it was trained on 10 years
of accepted CVs (predominantly male). Amazon withdrew it in 2018 without deployment.

---

## 1. Domain context

AI Act Category 4 (employment and workers management).
A black-box CV-screening model shortlists candidates for interview. The deployer
(employer) has only `predict_proba` access.

**Failure mode:** `gender_proxy_hiring` — gender-correlated proxy features
(technical keyword density, sports captaincy score, all-women-college affiliation,
male-employee referral network strength) contaminate the shortlisting decision.
Merit features are independent of gender by construction; the proxy signal enters
through vocabulary and social network artefacts in historical training data.

## 2. Dataset design (three criteria)

### Feature fidelity

| Feature | Type | Role |
|---|---|---|
| years_experience | float [0–25] | Merit (legitimate) |
| num_skills_matched | float [0–1] | Merit (legitimate) |
| education_level | ordinal {0,1,2,3} | Merit (legitimate) |
| portfolio_score | float [0–1] | Merit (legitimate) |
| interview_score | float [0–1] | Merit (legitimate) |
| **technical_keyword_score** | float [0–1] | **Proxy — male-coded terminology density** |
| **sports_captain_flag** | float [0–1] | **Proxy — sports/athletic leadership (male-coded)** |
| **all_women_college** | float [0–1] | **Proxy — women's-institution affiliation (penalised)** |
| **referral_from_male_employee** | float [0–1] | **Proxy — male-network endorsement strength** |

`gender` (0=male, 1=female) and `candidate_quality` are latent construction variables —
not in FEATURE_NAMES and not used by the model.

### Structural fidelity

- A shared `candidate_quality` latent factor drives all five merit features with 70%
  weight (30% idiosyncratic noise), inducing realistic inter-feature correlations:
  strong candidates score high across all merit dimensions simultaneously.
- `gender` drives all four proxy features but has zero direct influence on merit.
  Male CVs score higher on `technical_keyword_score`, `sports_captain_flag`, and
  `referral_from_male_employee`; female CVs score higher on `all_women_college`.
- The latent factor creates a sufficient spread of high- and low-quality candidates
  for the model to learn, producing an adequate far-from-threshold pool.

### Controlled ground truth

| Coded effect | Target magnitude |
|---|---|
| Proxy signal absolute weight (dataset) | ~25% |
| Male shortlist rate | ~48–49% |
| Female shortlist rate | ~38–39% |
| Gap (proxy-driven) | ~10pp |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 40 |
| Mean flip-rate | 0.085 ± 0.131 | 0.000 ± 0.000 |
| Mean proxy attribution | 0.355 ± 0.113 | 0.142 ± 0.062 |
| Mean legitimate ratio | 0.645 | 0.858 |
| Mean envelope distance | 1.890 ± 0.317 | 2.350 ± 0.360 |
| Mean attr. stability (σ rank) | 0.088 ± 0.065 | 0.038 ± 0.055 |
| Flagged (%) | 45.0% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 3.43e-07 → DISCRIMINATES (p < 0.01)
- Envelope distance near vs. far: p = 1.00e+00 → does not discriminate (expected: gender proxy bias is a within-manifold failure — female and male CVs occupy the same input space; the model's contamination is internal to its attribution, not a distributional shift in the inputs)
- Attr. stability near vs. far: p = 1.61e-05 → DISCRIMINATES (p < 0.01)

## 4. Evidence snippet (7 signals)

```json
{
  "case_id": "amazon_hiring-near-0000",
  "demonstrator": "amazon_hiring",
  "failure_mode": "gender_proxy_hiring",
  "annex_iii_category": "cat_4",
  "output_score": 0.5155,
  "threshold": 0.5,
  "margin_to_threshold": 0.0155,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.6444,
    "proxy_attribution_total": 0.3556,
    "proxy_detail": {
      "technical_keyword_score": 0.2282,
      "sports_captain_flag": 0.0623,
      "all_women_college": 0.0583,
      "referral_from_male_employee": 0.0069
    },
    "top_features": ["education_level", "technical_keyword_score",
                     "years_experience", "num_skills_matched"],
    "status": "OK"
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 1.9642,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.28,
    "status": "OK"
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.0,
    "status": "OK"
  },
  "sustainability_impact": {
    "dimension": "workforce_diversity",
    "at_risk_group": "female_candidates",
    "indicator": "gender_proxy_shortlist_rate",
    "value": 0.3556,
    "status": "FLAG"
  },
  "human_ai_teaming": {
    "dimension": "hiring_oversight",
    "escalation_level": "recommended_review",
    "trigger": "near_threshold_with_gender_proxy_attribution",
    "human_override_available": true,
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

`sustainability_impact` is FLAG (proxy > 0.25): 35.6% of this shortlisting decision
traces to gender-correlated proxies — vocabulary and social network artefacts, not
job-relevant merit. `human_ai_teaming` escalates to `recommended_review`: a
near-threshold decision with gender proxy attribution warrants human confirmation
before acting on the shortlist.

## 5. Interpretation

The instrument **recovers the coded gender proxy contamination**: proxy attribution
near-threshold ≈ 35.5% (coded ~25% population-level weight). The near-threshold
reading exceeds the population target because borderline cases are precisely those
where legitimate features are ambiguous — neither strongly approving nor strongly
denying — and gender proxies carry disproportionate influence at the margin. This
amplification effect is structurally expected: the coded weight (25%) describes the
proxy's contribution averaged over all decisions; the instrument's per-case reading
at the threshold (35.5%) reveals where the proxy is most consequential.

**Flip-rate discriminates** (p = 3.43e-07): near-threshold shortlisting decisions
are significantly more fragile than far-from-threshold ones. Small perturbations to
proxy feature values — changes that a candidate could make by rewording a CV —
reverse the shortlisting outcome in ~8.5% of cases on average.

**Attr. stability also discriminates** (p = 1.61e-05): near-threshold cases show
less stable feature rank orders under repeated attribution, consistent with the proxy
signal introducing ambiguity at the decision boundary. When technical_keyword_score
appears in the top features for a shortlisted candidate (as in the snippet above),
the ranking is sensitive to small input perturbations — a diagnostic that the
attribution structure is not firmly grounded in merit.

**Envelope does not discriminate** (expected): gender proxy bias is a within-manifold
failure. Female and male candidates' CVs occupy the same 9-dimensional input space;
the failure is not that female CVs are out-of-distribution, but that the model has
learned to weight male-coded signals for in-distribution inputs. The envelope's
silence confirms the inputs are operationally valid; the failure is internal to the
model's learned representation.

The dataset shortlist rates confirm the proxy impact: male candidates 48.4%,
female candidates 37.9% (10.5pp gap). The instrument makes this aggregate disparity
visible at the individual decision level, per-case, enabling the deployer to audit
specific shortlisting outcomes rather than relying solely on aggregate fairness metrics.
