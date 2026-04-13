# Cat. 7 — Migration: Home Office Visa Nationality Proxy Streaming

AI Act Category 7 covers AI systems used in migration, asylum, and border
management. This demonstrator instruments the failure mode of nationality
proxy streaming: an opaque legacy risk score — itself a black-box — is
delegated as a sub-system of the current visa approval model, creating a
black-box-inside-a-black-box structure.

Inspired by the Home Office UK algorithmic streaming tool (2015–2020),
which classified visa applications into red/amber/green risk streams.
The Foxglove / JCWI 2020 audit found that nationality was the dominant
factor, systematically downgrading applicants from certain countries
regardless of individual merit. The Home Office withdrew it in August 2020.

The black-box-inside-a-black-box property is structurally analogous to
HELOC (Cat. 5): just as the credit model delegates part of its decision
to an opaque bureau score, the visa model delegates part of its decision
to an opaque nationality streaming score whose internal logic is not
auditable by the current system's operators.

---

## 1. Domain context

AI Act Category 7 (migration, asylum, and border management).
A visa approval model processes individual applications. The deployer
(Home Office caseworker / automated processing system) has only
`predict_proba` access.

**Failure mode:** `nationality_proxy_streaming` — an opaque legacy streaming
score (`nationality_risk_score`) dominates the visa approval decision.
The score aggregates nationality-tier information through internal logic
the current deployer cannot inspect. Applicants from high-risk-tier
countries receive systematically higher scores regardless of their
individual visa compliance history, income, or sponsorship strength.

**Foxglove/JCWI 2020 property:** The audit found that amber/red streaming
decisions did not trigger human caseworker review. The snippet this
instrument produces is the per-case attribution that was absent from those
decisions — making the opaque score's individual contribution visible
without requiring access to its internal logic.

## 2. Dataset design (three criteria)

### Feature fidelity

| Feature | Type | Role |
|---|---|---|
| years_in_country | float [0–20] | Residence ties (legitimate) |
| income_level | float [0–1] | Financial eligibility (legitimate) |
| sponsor_type | ordinal {0,1,2} | Sponsorship strength (legitimate) |
| prior_visa_compliance | float [0–1] | Track record (legitimate) |
| purpose_of_travel | ordinal {0,1,2,3} | Intent/ties (legitimate) |
| **nationality_risk_score** | float [5–95] | **Opaque legacy streaming score (proxy — black-box inside black-box)** |
| **country_of_origin_tier** | ordinal {0,1,2} | **Nationality tier (proxy — 0=low-risk, 2=high-risk)** |
| **travel_history_flag** | binary {0,1} | **Complex travel history (proxy — structurally higher for high-tier)** |

`applicant_legitimacy` is a latent construction variable — not in FEATURE_NAMES
and not used by the model.

### Structural fidelity

- A shared `applicant_legitimacy` latent factor drives all five legitimate
  features with 70% weight (30% idiosyncratic noise), creating correlated
  applicant profiles: strong applicants score high across all legitimacy
  dimensions simultaneously.
- `nationality_risk_score` is generated from `country_of_origin_tier` (25%
  weight) + partial legitimacy component (20% weight) + large opaque variance
  (σ=0.22). The large opaque component ensures the model must rely on the
  score directly rather than reconstructing it from the observable tier —
  this models the real-world situation where the legacy system's internal
  logic is irreversible.
- `country_of_origin_tier` drives `travel_history_flag` (higher tier →
  more complex travel history: 10% for tier-0, 35% for tier-2), creating
  structural correlation among proxy features.
- Approval rate gap: tier-0 50.6%, tier-2 44.6% (~6pp gap driven by
  nationality_risk_score's dominance of the proxy signal).

### Controlled ground truth

| Coded effect | Target magnitude |
|---|---|
| nationality_risk_score coded weight | ~30% of combined signal |
| nationality_risk_score attribution near-threshold | ~28–32% |
| Cases where NRS attribution > 30% (nationality-dominated) | ~25% |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 40 |
| Mean flip-rate | 0.025 ± 0.076 | 0.000 ± 0.000 |
| Mean proxy attribution | 0.365 ± 0.178 | 0.216 ± 0.086 |
| Mean NRS attribution | 0.308 ± 0.177 | 0.203 ± 0.084 |
| Mean legitimate ratio | 0.635 | 0.784 |
| Mean envelope distance | 1.175 ± 0.192 | 1.403 ± 0.346 |
| Mean attr. stability (σ rank) | 0.079 ± 0.069 | 0.032 ± 0.046 |
| Flagged (%) | 57.5% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 1.87e-13 → DISCRIMINATES (p < 0.01)
- Envelope distance near vs. far: p = 1.00e+00 → does not discriminate (expected: nationality proxy streaming is a within-manifold failure — applicants of all nationalities appear in the same 8-dimensional input space; the failure is internal to the model's use of the opaque score, not a distributional shift in inputs)
- Attr. stability near vs. far: p = 1.66e-04 → DISCRIMINATES (p < 0.01)

## 4. Evidence snippet (7 signals)

```json
{
  "case_id": "home_office_visa-near-0003",
  "demonstrator": "home_office_visa",
  "failure_mode": "nationality_proxy_streaming",
  "annex_iii_category": "cat_7",
  "output_score": 0.5321,
  "threshold": 0.5,
  "margin_to_threshold": 0.0321,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.5745,
    "proxy_attribution_total": 0.4255,
    "proxy_detail": {
      "nationality_risk_score": 0.3612,
      "country_of_origin_tier": 0.0444,
      "travel_history_flag": 0.0199
    },
    "top_features": ["nationality_risk_score", "purpose_of_travel",
                     "prior_visa_compliance", "sponsor_type"],
    "status": "CONTESTED"
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 1.2369,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.06,
    "status": "OK"
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.1,
    "status": "OK"
  },
  "sustainability_impact": {
    "dimension": "migration_equity",
    "at_risk_group": "high_tier_nationalities",
    "indicator": "nationality_proxy_rejection_rate",
    "value": 0.3612,
    "status": "FLAG"
  },
  "human_ai_teaming": {
    "dimension": "caseworker_oversight",
    "escalation_level": "recommended_review",
    "trigger": "near_threshold_nationality_proxy_streaming",
    "human_override_available": true,
    "note": "Foxglove/JCWI 2020: no human review for amber/red streaming decisions",
    "status": "OK"
  },
  "action_verdict": "PROCEED WITH FLAG"
}
```

`constraint_enforcement` is CONTESTED (legitimate_ratio 0.57 < 0.60): the
nationality_risk_score appears as the top feature, contributing 36.1% of
attribution — more than any individual legitimate criterion. The score is
not internally decomposable; the caseworker sees only this aggregate value.
`sustainability_impact` is FLAG: 36.1% of this visa decision traces to the
opaque nationality-tier score. `human_ai_teaming` escalates to
`recommended_review`: under the Foxglove/JCWI-identified absence of
caseworker review for amber/red streams, this case would have been decided
without human oversight.

## 5. Interpretation

The instrument **recovers the coded score-on-score opacity**: NRS attribution
near-threshold ≈ 30.8% (target ~30%, coded weight ~30%). Unlike HELOC
(Cat. 5), where near-threshold dilution reduced the measured attribution to
14.8% of the 20.3% coded target, home_office_visa achieves near-target
recovery. The difference is structural: HELOC has 12 legitimate features
that collectively dilute the bureau score's attribution share; home_office
has 5 legitimate features, giving the opaque score more attributional space.

**The HELOC analogy — and its limit.** Both HELOC and home_office_visa
instrument the black-box-inside-a-black-box failure mode: the deployer's
model delegates to a sub-system whose internal logic is opaque. In HELOC,
the deployer is a credit lender who cannot see what drives the bureau's
score. In home_office_visa, the deployer is a government caseworker who
cannot see what drove the legacy streaming system's nationality risk score.
The instrument surfaces the delegation in both cases without requiring
access to the sub-system's internals. What distinguishes the two domains
is the nature of the harm: credit dilution is financial; nationality
streaming produces systematic barriers to migration that disproportionately
affect citizens of specific countries.

**`nationality_risk_score` appears in top_features.** In the snippet,
`nationality_risk_score` is the single highest-attribution feature, ranking
above all individual legitimate criteria. This is not a spurious correlation:
the score was actively used by the prior streaming system as a primary
classification criterion, and the current model has inherited its influence
through training data that reflected its decisions.

**Flip-rate discriminates** (p = 1.87e-13): near-threshold visa decisions
are significantly more fragile than far-from-threshold ones. A 6% flip-rate
in the snippet means that 3 of 50 bounded perturbations of this applicant's
features would reverse the decision — fragility that is invisible to the
caseworker receiving only the point score.

**Attr. stability discriminates** (p = 1.66e-04): near-threshold cases show
less stable feature rank orders, consistent with the opaque score introducing
ambiguity at the decision boundary. When the nationality_risk_score tips a
borderline case, the feature ranking is sensitive to small perturbations —
a sign that the decision is not firmly grounded in the individual's
legitimate credentials.

**Envelope does not discriminate** (expected): all nationalities appear in
the same 8-dimensional input space; the failure is not that high-tier
applicants are out-of-distribution, but that the model has incorporated an
opaque score that encodes nationality as a risk factor for in-distribution
inputs. The envelope's silence confirms the inputs are operationally valid;
the contamination is internal to the model's learned weighting of the score.

**Auditability without resolution.** Neither the current deployer nor the
instrument can decompose what drives `nationality_risk_score` — that requires
access to the legacy system. The snippet surfaces the delegation: 36.1% of
this decision traces to an opaque sub-system. This is auditable even if
not resolvable; the deployer can flag, escalate, and demand vendor
transparency without possessing it. This was precisely what Foxglove/JCWI
demonstrated was absent from the original streaming decisions.
