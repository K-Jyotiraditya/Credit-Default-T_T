# Project 7 — Theory & Deep Dive: Credit Scorecard Development

The full scorecard workflow and the *why* behind each step. Read alongside the
code; sections map to `woe.py`, `scorecard.py`, `model.py`, `evaluate.py`, and
`economics.py`.

---

## 1. The problem (and why a random split is fine)

Estimate `PD = P(default | features)`. The data is **cross-sectional** (one row
per client, no time order), so a stratified random split is correct and leak-free
— there is no past/future to respect, unlike the price series in projects 6 & 9.
We stratify so train and test share the ~22% base rate.

---

## 2. Weight of Evidence & Information Value (`woe.py`)

A scorecard does not feed raw features to the model — it recodes each into its
**Weight of Evidence**:

```
WOE_b = ln( (good_b / total_good) / (bad_b / total_bad) )
```

over bins `b` (quantile bins for continuous features, one bin per value for coded
ones). WOE is monotone in risk, immune to outliers, handles missing as its own
bin, and linearises the relationship so a *logistic* model fits it well. A small
Laplace count (`smoothing`) keeps WOE finite when a bin has zero goods or bads.

**Information Value** ranks features:

```
IV = Σ_b (good_b/total_good − bad_b/total_bad) · WOE_b
```

with the usual bands (<0.02 useless, 0.1–0.3 medium, 0.3–0.5 strong, **>0.5
"suspicious" → check for leakage**). Here `PAY_0` (last month's repayment status)
has IV≈0.89 — flagged, but legitimate: it is known *before* the decision and is
genuinely the strongest predictor of next-month default.

`WOEBinner` is a scikit-learn transformer, so it is **refit inside every CV fold
and every calibration split** — WOE fit on the whole sample would leak the target
through the binning.

---

## 3. The points scorecard (`scorecard.py`)

Fit a logistic regression on the WOE features, then scale log-odds to points
(Siddiqi): with `factor = PDO/ln2` and `offset = base − factor·ln(base_odds)`,

```
points(feature=j, bin=b) = −(WOE_jb·coef_j + intercept/n)·factor + offset/n
```

so points sum to a score where **+PDO points doubles the good:bad odds** (here
PDO=20, base 600 at 50:1). The score inverts back to a PD (`score_to_pd`). This is
the artifact a credit officer reads and a regulator audits.

---

## 4. Calibration — because PD is a probability, not a ranking (`model.py`)

Ranking metrics (AUC/KS) don't care if probabilities are *right*, only *ordered*.
But a PD feeds **expected loss** `EL = PD·LGD·EAD` and IFRS-9 / capital, so it must
be calibrated. Two choices matter:

* Use the **natural class prior** (no `class_weight="balanced"`), which would
  inflate PDs and destroy calibration for a marginal ranking gain.
* Wrap the model in **`CalibratedClassifierCV` (isotonic)**, fit by internal CV,
  and verify the calibration curve sits on the diagonal (the Brier score barely
  moves here precisely because the natural-prior model is already near-calibrated).

---

## 5. Validation: cross-validation + stability (`evaluate.py`)

* **Stratified K-fold CV** reports AUC/Gini/KS as **mean ± std**, not a single
  noisy split — the scorecard's `0.771 ± 0.006` shows the estimate is stable.
* **PSI** (Population Stability Index) `= Σ (a−e)·ln(a/e)` over fixed score bins
  measures distribution drift between the development and a later sample
  (<0.10 stable, >0.25 material). It is *the* model-monitoring trigger in
  production; here train-vs-test PSI ≈ 0.002 confirms a stable score.
* **KS** uses a tie-grouped implementation so a flat-score model correctly scores 0.

---

## 6. Economics: from PD to a decision (`economics.py`)

Discrimination is necessary but not sufficient — the model has to make money.
Approving when `PD ≤ cut-off`, a good account earns `revenue_rate·EAD` and a
default costs `LGD·EAD` (EAD proxied by the credit limit). Sweeping the cut-off
traces the **profit curve**; its peak is the profit-maximising policy, which sits
*below* any accuracy-optimal threshold because the asymmetric loss (LGD ≫ margin)
makes false approvals expensive. The approval-rate-vs-bad-rate panel is the
classic origination trade-off.

**Reason codes** (`reason_codes`) fall out of the scorecard for free: for a
declined applicant, the features costing the most points versus their best
attainable bin are the **adverse-action reasons** lenders must legally disclose
(FCRA/ECOA).

---

## 7. Fair lending (`data.py`)

ECOA prohibits credit decisions on **sex, marital status, and age**. We drop
`SEX`, `MARRIAGE`, `AGE` (and `EDUCATION` as a proxy) before modelling, so neither
the score nor its reason codes depend on protected attributes. Dropping them cost
~0.004 AUC — they carried almost no signal beyond the behavioural variables, so
compliance here is nearly free. This is the difference between a homework model
and one that could face a regulator.

---

## 8. Limitations

* **Reject inference.** The data is *accepted* applicants only; a production
  scorecard must infer the performance of rejected applicants (survivorship bias).
* **Single-period, through-the-cycle PD.** No macro conditioning (PIT vs TTC), no
  vintage/seasoning.
* **Proxy economics.** `revenue_rate`, `LGD`, and EAD-as-limit are illustrative;
  real numbers come from the product P&L and recovery data.
* **One stability sample.** PSI is shown train-vs-test; true monitoring needs an
  out-of-time sample as the population drifts.
