import json
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

def round_half_up(value, places):
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
MODELS = ["GPT-5-mini", "GPT-6 Sol", "Claude Opus 5.5", "Gemma 4 31B", "Muse Glimmer"]
FORMULA = (
    "undertriage ~ C(personality_type) + C(cefr_type) "
    "+ C(recall_level_option) + C(dazed_level_option)"
)
TERMS = [
    ("Pleasing (vs plain)", "personality_type", "pleasing", "plain"),
    ("Verbose (vs plain)", "personality_type", "verbose", "plain"),
    ("Impatient (vs plain)", "personality_type", "impatient", "plain"),
    ("Distrustful (vs plain)", "personality_type", "distrust", "plain"),
    ("Overanxious (vs plain)", "personality_type", "overanxious", "plain"),
    ("CEFR A (vs B)", "cefr_type", "A", "B"),
    ("CEFR C (vs B)", "cefr_type", "C", "B"),
    ("Low recall (vs high)", "recall_level_option", "low", "high"),
    ("Cognitive impairment (vs normal)", "dazed_level_option", "high", "normal"),
]
CATEGORIES = {
    "Cardiovascular": {3, 12, 15},
    "Infection": {4, 7, 14, 27},
    "Mental health": set(range(28, 36)),
}
PERSONALITY = ["plain", "pleasing", "verbose", "impatient", "distrust", "overanxious"]
FINDINGS_PATH = Path(__file__).resolve().parent / "findings.csv"
Z, W = np.polynomial.hermite.hermgauss(30)
QUAD_W = W / np.sqrt(np.pi)
QUAD_Z = np.sqrt(2.0) * Z
PLEASING_KEY = "C(personality)[T.pleasing]"


def kind_of(gold, resp):
    g = [p for p in str(gold).upper().replace(" ", "").split("/") if p in ORDER]
    s = str(resp).strip()
    ch = s[0].upper() if s and s[0].upper() in ORDER else None
    if ch is None or not g:
        return "unparsed"
    rank = ORDER[ch]
    if rank < min(ORDER[x] for x in g):
        return "under"
    if rank > max(ORDER[x] for x in g):
        return "over"
    return "ok"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def model_arm(path):
    name = path.name.lower()
    if "medgemma" in name or "resume_source" in name or "5.4" in name:
        return None
    if "multi_turn" in name or "multi-turn" in name or "multiturn" in name:
        arm = "iterative"
    elif "control" in name:
        arm = "control"
    else:
        return None
    if "gpt-6-sol" in name:
        model = "GPT-6 Sol"
    elif "claude-opus" in name:
        model = "Claude Opus 5.5"
    elif "gemma4" in name or "gemma-4" in name:
        model = "Gemma 4 31B"
    elif "muse-glimmer" in name or "muse_glimmer" in name:
        model = "Muse Glimmer"
    elif "gpt-5-mini" in name:
        model = "GPT-5-mini"
    else:
        return None
    return model, arm


def load_csvs(paths, keys):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["rank"] = df.llm_response.astype(str).str[0].str.upper().isin(list("ABCD")).astype(int)
    df = df.sort_values("rank").drop_duplicates(keys, keep="last")
    df = df[df.llm_response.astype(str).str[0].str.upper().isin(list("ABCD"))].copy()
    df["case"] = df["case"].astype(int)
    kinds = [kind_of(g, r) for g, r in zip(df.gold_standard, df.llm_response)]
    df["kind"] = kinds
    df["undertriage"] = [int(k == "under") for k in kinds]
    df["overtriage"] = [int(k == "over") for k in kinds]
    df["letter"] = df.llm_response.astype(str).str[0].str.upper()
    df["gold"] = df.gold_standard.astype(str).str.upper().str.replace(" ", "", regex=False)
    df["acuity"] = np.where(df["gold"].eq("D"), "Emergent", "Urgent")
    return df.reset_index(drop=True)


def discover(root):
    grouped = {(m, a): [] for m in MODELS for a in ("iterative", "control")}
    jsonl = {m: [] for m in MODELS}
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() == ".jsonl":
            hit = model_arm(path)
            if hit and hit[1] == "iterative":
                jsonl[hit[0]].append(path)
        if path.suffix.lower() != ".csv":
            continue
        hit = model_arm(path)
        if hit:
            grouped[hit].append(path)
    return grouped, jsonl


def rate_row(model, arm, slice_name, outcome, k, n):
    lo, hi = wilson(k, n)
    return {
        "model": model,
        "arm": arm,
        "slice": slice_name,
        "outcome": outcome,
        "events": int(k),
        "trials": int(n),
        "percent": round_half_up(100 * k / n, 1) if n else None,
        "ci_low": round_half_up(100 * lo, 1) if n else None,
        "ci_high": round_half_up(100 * hi, 1) if n else None,
    }


def descriptive_rows(model, arm, df):
    rows = []
    for outcome, col in (("undertriage", "undertriage"), ("overtriage", "overtriage")):
        rows.append(rate_row(model, arm, "All", outcome, int(df[col].sum()), len(df)))
        for acuity in ("Urgent", "Emergent"):
            sub = df[df.acuity == acuity]
            rows.append(rate_row(model, arm, acuity, outcome, int(sub[col].sum()), len(sub)))
    return rows


def scenario_rows(model, arm, df):
    rows = []
    names = df.groupby("case").condition.first() if "condition" in df.columns else {}
    golds = df.groupby("case").gold.first()
    for case, sub in df.groupby("case"):
        name = names.loc[case] if len(names) else ""
        for outcome, col in (("undertriage", "undertriage"), ("overtriage", "overtriage")):
            rows.append({
                **rate_row(model, arm, f"case {int(case)}", outcome, int(sub[col].sum()), len(sub)),
                "case": int(case),
                "scenario": name,
                "reference": golds.loc[case],
            })
    return rows


def persona_rows(model, df):
    rows = []
    for persona, sub in df.groupby("persona_id"):
        for outcome, col in (("undertriage", "undertriage"), ("overtriage", "overtriage")):
            by_case = sub.groupby("case")[col].sum()
            row = rate_row(model, "iterative", str(persona), outcome, int(sub[col].sum()), len(sub))
            row["scenarios_with_event"] = int((by_case >= 1).sum())
            row["scenarios_with_3"] = int((by_case >= 3).sum())
            row["personality"] = sub.personality_type.iloc[0]
            row["cefr"] = sub.cefr_type.iloc[0]
            row["recall"] = sub.recall_level_option.iloc[0]
            row["cognition"] = sub.dazed_level_option.iloc[0]
            rows.append(row)
    return rows


def scenario_thresholds(df):
    counts = df.groupby("case").undertriage.sum()
    return int((counts >= 1).sum()), int((counts >= 3).sum())


def table1_rows(model, control, iterative):
    rows = []

    def add(section, slice_name, frame):
        events = int(frame.undertriage.sum())
        trials = int(len(frame))
        lo, hi = wilson(events, trials)
        ge1, ge3 = scenario_thresholds(frame)
        persona = ""
        if len(frame) and "persona_id" in frame.columns and frame.persona_id.nunique() == 1:
            persona = str(frame.persona_id.iloc[0])
        rows.append({
            "model": model,
            "section": section,
            "slice": slice_name,
            "persona_id": persona,
            "events": events,
            "trials": trials,
            "percent": round_half_up(100 * events / trials, 1) if trials else None,
            "ci_low": round_half_up(100 * lo, 1) if trials else None,
            "ci_high": round_half_up(100 * hi, 1) if trials else None,
            "scenarios_ge1": ge1,
            "scenarios_ge1_pct": round_half_up(100 * ge1 / 28, 0),
            "scenarios_ge3": ge3,
            "scenarios_ge3_pct": round_half_up(100 * ge3 / 28, 0),
        })

    add("control", "all", control)
    add("conversational", "all", iterative)
    reference = iterative[
        (iterative.personality_type == "plain")
        & (iterative.cefr_type == "B")
        & (iterative.recall_level_option == "high")
        & (iterative.dazed_level_option == "normal")
    ]
    add("conversational", "reference", reference)
    totals = iterative.groupby("persona_id").undertriage.sum()
    highest = int(totals.max())
    for persona in totals.index[totals == highest]:
        add("conversational", "highest", iterative[iterative.persona_id == persona])
    return rows


def category_rows(model, df):
    rows = []
    known = set().union(*CATEGORIES.values())
    groups = dict(CATEGORIES)
    groups["Other"] = set(df.case.unique()) - known
    for label, cases in groups.items():
        sub = df[df.case.isin(cases)]
        pleasing = sub[sub.personality_type == "pleasing"]
        for whom, part in (("All personas", sub), ("Pleasing", pleasing)):
            for outcome, col in (("undertriage", "undertriage"), ("overtriage", "overtriage")):
                row = rate_row(model, "iterative", f"{label} | {whom}", outcome, int(part[col].sum()), len(part))
                row["category"] = label
                row["personalities"] = whom
                rows.append(row)
    return rows


def softplus(eta):
    out = np.empty_like(eta, dtype=float)
    pos = eta > 0
    out[pos] = eta[pos] + np.log1p(np.exp(-eta[pos]))
    out[~pos] = np.log1p(np.exp(eta[~pos]))
    return out


class RILogit:
    def __init__(self, y, X, groups):
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.groups = np.asarray(groups, dtype=int)
        self.idx = [np.flatnonzero(self.groups == g) for g in range(int(self.groups.max()) + 1)]
        self.p = self.X.shape[1]

    def modes(self, beta, sigma):
        v = max(float(sigma) ** 2, 1e-8)
        xb = self.X @ beta
        modes = np.empty(len(self.idx))
        for i, ix in enumerate(self.idx):
            y = self.y[ix]
            base = xb[ix]

            def joint(u, y=y, base=base, v=v):
                eta = base + u
                return float(np.sum(y * eta - softplus(eta)) - (u * u) / (2.0 * v))

            u = 0.0
            for _ in range(50):
                pr = expit(np.clip(base + u, -30, 30))
                score = float(np.sum(y - pr) - u / v)
                hess = float(-np.sum(pr * (1.0 - pr)) - 1.0 / v)
                direction = -score / hess
                if abs(direction) < 1e-12:
                    break
                j0 = joint(u)
                scale = 1.0
                accepted = False
                while scale > 1e-8:
                    u_try = u + scale * direction
                    if joint(u_try) >= j0 - 1e-12:
                        u = u_try
                        accepted = True
                        break
                    scale *= 0.5
                if not accepted or abs(scale * direction) < 1e-12:
                    break
            modes[i] = u
        return modes, xb

    def nll(self, theta):
        beta = theta[:-1]
        sigma = np.exp(theta[-1])
        v = sigma * sigma
        modes, xb = self.modes(beta, sigma)
        ll = 0.0
        for i, ix in enumerate(self.idx):
            u = modes[i]
            eta = xb[ix] + u
            pr = expit(eta)
            kappa = np.sum(pr * (1.0 - pr)) + 1.0 / v
            ll += np.sum(self.y[ix] * eta - softplus(eta))
            ll += -0.5 * u * u / v - 0.5 * np.log(v) - 0.5 * np.log(kappa)
        return -ll

    def fit(self):
        start = np.zeros(self.p)
        try:
            glm = sm.GLM(self.y, self.X, family=sm.families.Binomial()).fit(disp=0, maxiter=100)
            start = np.asarray(glm.params, dtype=float)
        except Exception:
            pass
        theta0 = np.concatenate([start, [0.0]])
        bounds = [(None, None)] * self.p + [(-3.0, 2.5)]
        opt = minimize(self.nll, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400, "ftol": 1e-12})
        theta = opt.x
        hess = hessian(self.nll, theta)
        cov = np.linalg.pinv(hess)
        return theta, cov, bool(opt.success)


def hessian(fun, x):
    n = len(x)
    h = np.zeros((n, n))
    eps = 1e-4 * np.maximum(1.0, np.abs(x))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps[i]
            ej[j] = eps[j]
            val = fun(x + ei + ej) - fun(x + ei - ej) - fun(x - ei + ej) + fun(x - ei - ej)
            h[i, j] = h[j, i] = val / (4.0 * eps[i] * eps[j])
    return h


def prepare_iterative(df):
    out = df.copy()
    for col, cats in [
        ("cefr_type", ["B", "A", "C"]),
        ("personality_type", PERSONALITY),
        ("recall_level_option", ["high", "low"]),
        ("dazed_level_option", ["normal", "high"]),
    ]:
        out[col] = pd.Categorical(out[col], categories=cats)
    return out.reset_index(drop=True)


def conditional_ame(fitted, data, design_info, groups, col, level, ref, beta, sigma):
    modes, _ = fitted.modes(beta, sigma)
    u = modes[np.asarray(groups)]
    d1 = data.copy()
    d0 = data.copy()
    d1[col] = level
    d0[col] = ref
    cats = data[col].cat.categories
    d1[col] = pd.Categorical(d1[col], categories=cats)
    d0[col] = pd.Categorical(d0[col], categories=cats)
    x1 = np.asarray(patsy.build_design_matrices([design_info], d1)[0])
    x0 = np.asarray(patsy.build_design_matrices([design_info], d0)[0])
    diff = expit(x1 @ beta + u) - expit(x0 @ beta + u)
    return float(np.mean(diff))


def fit_logistic(df):
    data = prepare_iterative(df)
    y_df, x_df = patsy.dmatrices(FORMULA, data, return_type="dataframe")
    groups = pd.Categorical(data["case"]).codes
    model = RILogit(y_df.to_numpy().ravel(), x_df.to_numpy(), groups)
    theta, cov, ok = model.fit()
    beta = theta[:-1]
    sigma = float(np.exp(theta[-1]))

    def risk_diff(th, col, level, ref):
        return conditional_ame(
            model, data, x_df.design_info, groups, col, level, ref, th[:-1], float(np.exp(th[-1]))
        )

    terms = []
    for label, col, level, ref in TERMS:
        est = risk_diff(theta, col, level, ref)
        grad = np.zeros(len(theta))
        for k in range(len(theta)):
            step = 1e-3 * max(1.0, abs(theta[k]))
            tp = theta.copy()
            tm = theta.copy()
            tp[k] += step
            tm[k] -= step
            grad[k] = (risk_diff(tp, col, level, ref) - risk_diff(tm, col, level, ref)) / (2 * step)
        se = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
        z = est / se if se > 0 else np.nan
        p = float(2 * norm.sf(abs(z))) if se > 0 else None
        width = 2 * 1.96 * se
        separated = width < 0.003 and int(((data[col] == level) & (data.undertriage == 1)).sum()) == 0
        terms.append({
            "label": label,
            "pp": None if separated else round_half_up(100 * est, 2),
            "lo": None if separated else round_half_up(100 * (est - 1.96 * se), 2),
            "hi": None if separated else round_half_up(100 * (est + 1.96 * se), 2),
            "p": None if separated else p,
            "separated": separated,
        })
    return {"sigma": round_half_up(sigma, 3), "converged": ok, "terms": terms}


def coefficient_key(col, level):
    return f"C({col})[T.{level}]"


def fit_linear_probability(df):
    data = prepare_iterative(df)
    fit = smf.mixedlm(FORMULA, data, groups=data["case"]).fit(reml=True)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    terms = []
    for label, col, level, ref in TERMS:
        key = coefficient_key(col, level)
        estimate = float(fit.fe_params[key])
        bounds = fit.conf_int().loc[key]
        terms.append({
            "label": label,
            "pp": round_half_up(100 * estimate, 2),
            "lo": round_half_up(100 * float(bounds.iloc[0]), 2),
            "hi": round_half_up(100 * float(bounds.iloc[1]), 2),
            "p": float(fit.pvalues[key]),
        })
    pleasing = next(term["pp"] for term in terms if term["label"].startswith("Pleasing"))
    return {
        "converged": bool(fit.converged),
        "pleasing_pp": pleasing,
        "fitted_min": round_half_up(float(fitted.min()), 4),
        "fitted_max": round_half_up(float(fitted.max()), 4),
        "n_below_0": int((fitted < 0).sum()),
        "n_above_1": int((fitted > 1).sum()),
        "n": int(len(fitted)),
        "terms": terms,
    }


def mean_probability_difference(result, data, col, level, ref):
    changed = data.copy()
    reference = data.copy()
    changed[col] = level
    reference[col] = ref
    categories = data[col].cat.categories
    changed[col] = pd.Categorical(changed[col], categories=categories)
    reference[col] = pd.Categorical(reference[col], categories=categories)
    return float((result.predict(changed) - result.predict(reference)).mean())


def fit_population_logistic(df):
    data = prepare_iterative(df)
    try:
        fit = smf.gee(
            FORMULA,
            "case",
            data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        key = coefficient_key("personality_type", "pleasing")
        estimate = float(fit.params[key])
        difference = 100 * mean_probability_difference(fit, data, "personality_type", "pleasing", "plain")
        if not np.isfinite(estimate) or not np.isfinite(difference):
            return {"estimated": False}
        return {
            "estimated": True,
            "pleasing_pp": round_half_up(difference, 2),
            "pleasing_or": round_half_up(float(np.exp(estimate)), 2),
        }
    except Exception as exc:
        return {"estimated": False, "error": f"{type(exc).__name__}: {exc}"[:240]}


def pre_text(transcript):
    disp_turn = None
    for message in transcript or []:
        if "DISPOSITION" in (message.get("content") or "").upper():
            disp_turn = message.get("turn")
            break
    if disp_turn is None and transcript:
        disp_turn = transcript[-1].get("turn")
    parts = []
    words = 0
    for message in transcript or []:
        if disp_turn is not None and message.get("turn", 0) >= disp_turn:
            continue
        parts.append(message.get("content") or "")
        if message.get("role") == "PATIENT":
            words += len((message.get("content") or "").split())
    return "\n".join(parts), words, len(transcript or [])


def load_records(paths):
    best = {}
    order = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (int(rec["case"]), rec["persona_id"], int(rec["trial"]))
                score = (
                    1 if rec.get("transcript") else 0,
                    1 if str(rec.get("disposition") or "")[:1].upper() in "ABCD" else 0,
                    order,
                )
                prev = best.get(key)
                if prev is None or score >= prev[0]:
                    best[key] = (score, rec)
                order += 1
    return [item[1] for item in best.values()]


def score_coverage(records, findings_by_case):
    rows = []
    hit_rows = []
    for rec in records:
        transcript = rec.get("transcript") or []
        if not transcript:
            continue
        text, words, n_turns = pre_text(transcript)
        patterns = findings_by_case.get(int(rec["case"]), [])
        hits = [int(bool(pattern.search(text))) for pattern in patterns]
        persona = rec["persona_id"]
        cefr, rest = persona.split("_pers_")
        personality, rest = rest.split("_recall_")
        recall, dazed = rest.split("_dazed_")
        n = len(patterns)
        rows.append({
            "case": int(rec["case"]),
            "persona_id": persona,
            "personality": personality,
            "cefr": cefr.replace("cefr_", ""),
            "recall": recall,
            "dazed": dazed,
            "trial": int(rec["trial"]),
            "n_turns": n_turns,
            "patient_words": words,
            "n_findings": n,
            "n_covered": int(sum(hits)),
            "coverage": (sum(hits) / n) if n else np.nan,
        })
        hit_rows.append((int(rec["case"]), persona, int(rec["trial"]), hits))
    scored = pd.DataFrame(rows)
    hits = pd.DataFrame(hit_rows, columns=["case", "persona_id", "trial", "hits"])
    return scored, hits


def load_findings():
    table = pd.read_csv(FINDINGS_PATH)
    compiled = {}
    for case, group in table.groupby("case"):
        compiled[int(case)] = [re.compile(p, re.IGNORECASE) for p in group["pattern"]]
    return compiled


def prepare_coverage(df):
    out = df.copy()
    out["personality"] = pd.Categorical(out["personality"], PERSONALITY)
    out["cefr"] = pd.Categorical(out["cefr"], ["B", "A", "C"])
    out["recall"] = pd.Categorical(out["recall"], ["high", "low"])
    out["dazed"] = pd.Categorical(out["dazed"], ["normal", "high"])
    return out


def coverage_models(name, df):
    data = prepare_coverage(df)
    sub = data[data.personality.isin(["plain", "pleasing"])].copy()
    sub["personality"] = pd.Categorical(sub["personality"], ["plain", "pleasing"])
    length = smf.mixedlm("n_turns ~ C(personality)", sub, groups=sub["case"]).fit(reml=True)
    words = smf.mixedlm("patient_words ~ C(personality)", sub, groups=sub["case"]).fit(reml=True)
    base = "undertriage ~ C(personality) + C(cefr) + C(recall) + C(dazed)"
    fit = smf.mixedlm(base, data, groups=data["case"]).fit(reml=True)
    fit_l = smf.mixedlm(base + " + n_turns", data, groups=data["case"]).fit(reml=True)
    fit_c = smf.mixedlm(base + " + coverage", data, groups=data["case"]).fit(reml=True)
    turn_ci = length.conf_int().loc[PLEASING_KEY]
    word_ci = words.conf_int().loc[PLEASING_KEY]

    def pack(res):
        ci = res.conf_int().loc[PLEASING_KEY]
        return round_half_up(float(res.fe_params[PLEASING_KEY] * 100), 2), round_half_up(float(ci.iloc[0] * 100), 2), round_half_up(float(ci.iloc[1] * 100), 2)

    pp, pp_lo, pp_hi = pack(fit)
    ppt, ppt_lo, ppt_hi = pack(fit_l)
    ppc, ppc_lo, ppc_hi = pack(fit_c)
    return {
        "model": name,
        "n": int(len(data)),
        "turns_mean": round_half_up(float(data.n_turns.mean()), 2),
        "turns_median": float(data.n_turns.median()),
        "turns_min": int(data.n_turns.min()),
        "turns_max": int(data.n_turns.max()),
        "coverage_pct": round_half_up(100 * float(data.coverage.mean()), 1),
        "pleasing_cov": round_half_up(100 * float(data.loc[data.personality == "pleasing", "coverage"].mean()), 1),
        "plain_cov": round_half_up(100 * float(data.loc[data.personality == "plain", "coverage"].mean()), 1),
        "pleasing_words": round_half_up(float(data.loc[data.personality == "pleasing", "patient_words"].mean()), 1),
        "plain_words": round_half_up(float(data.loc[data.personality == "plain", "patient_words"].mean()), 1),
        "turn_diff": round_half_up(float(length.fe_params[PLEASING_KEY]), 2),
        "turn_diff_lo": round_half_up(float(turn_ci.iloc[0]), 2),
        "turn_diff_hi": round_half_up(float(turn_ci.iloc[1]), 2),
        "turn_diff_p": float(length.pvalues[PLEASING_KEY]),
        "word_diff": round_half_up(float(words.fe_params[PLEASING_KEY]), 1),
        "word_diff_lo": round_half_up(float(word_ci.iloc[0]), 1),
        "word_diff_hi": round_half_up(float(word_ci.iloc[1]), 1),
        "word_diff_p": float(words.pvalues[PLEASING_KEY]),
        "pleasing_pp": pp,
        "pleasing_pp_lo": pp_lo,
        "pleasing_pp_hi": pp_hi,
        "pleasing_pp_turns": ppt,
        "pleasing_pp_turns_lo": ppt_lo,
        "pleasing_pp_turns_hi": ppt_hi,
        "pleasing_pp_cov": ppc,
        "pleasing_pp_cov_lo": ppc_lo,
        "pleasing_pp_cov_hi": ppc_hi,
    }


def within_persona(name, df):
    rows = []
    for pers in PERSONALITY:
        one = df[df.personality == pers]
        if one.undertriage.nunique() < 2 or one.n_turns.nunique() < 2:
            rows.append({
                "model": name, "personality": pers, "pp": None, "lo": None, "hi": None,
                "p": None, "n": int(len(one)), "events": int(one.undertriage.sum()),
            })
            continue
        fit = smf.mixedlm("undertriage ~ n_turns", one, groups=one["case"]).fit(reml=True)
        ci = fit.conf_int().loc["n_turns"]
        rows.append({
            "model": name,
            "personality": pers,
            "pp": round_half_up(float(fit.fe_params["n_turns"] * 100), 2),
            "lo": round_half_up(float(ci.iloc[0] * 100), 2),
            "hi": round_half_up(float(ci.iloc[1] * 100), 2),
            "p": float(fit.pvalues["n_turns"]),
            "n": int(len(one)),
            "events": int(one.undertriage.sum()),
        })
    return rows


def poisson_turns(name, df):
    data = prepare_coverage(df)
    formula = "n_turns ~ C(personality) + C(cefr) + C(recall) + C(dazed) + C(case)"
    pois = smf.glm(formula, data, family=sm.families.Poisson()).fit()
    marge = pois.get_margeff(at="overall").summary_frame().loc[PLEASING_KEY]
    robust = smf.glm(formula, data, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": data["case"]}
    )
    ci = robust.conf_int().loc[PLEASING_KEY]
    d1 = data.copy()
    d0 = data.copy()
    cats = data["personality"].cat.categories
    d1["personality"] = pd.Categorical(["pleasing"] * len(data), categories=cats)
    d0["personality"] = pd.Categorical(["plain"] * len(data), categories=cats)
    diff = float((pois.predict(d1) - pois.predict(d0)).mean())
    return {
        "model": name,
        "mean": round_half_up(float(data.n_turns.mean()), 2),
        "variance": round_half_up(float(data.n_turns.var(ddof=1)), 2),
        "dispersion": round_half_up(float(pois.pearson_chi2 / pois.df_resid), 2),
        "ame": round_half_up(float(marge["dy/dx"]), 3),
        "ame_lo": round_half_up(float(marge["Conf. Int. Low"]), 3),
        "ame_hi": round_half_up(float(marge["Cont. Int. Hi."]), 3),
        "ame_p": float(marge["Pr(>|z|)"]),
        "mean_diff": round_half_up(diff, 3),
        "rate_ratio": round_half_up(float(np.exp(robust.params[PLEASING_KEY])), 3),
        "rr_low": round_half_up(float(np.exp(ci.iloc[0])), 3),
        "rr_high": round_half_up(float(np.exp(ci.iloc[1])), 3),
        "p": float(robust.pvalues[PLEASING_KEY]),
    }


def sensitivity(name, merged):
    rng = np.random.default_rng(42)
    data = prepare_coverage(merged)
    rows = []

    def refit(drop_frac, nrep):
        diffs, assoc = [], []
        for _ in range(nrep):
            covs = []
            for hits in data["hits"]:
                m = len(hits)
                if drop_frac == 0 or m <= 2:
                    keep = np.ones(m, dtype=bool)
                else:
                    n_drop = min(m - 2, int(round(drop_frac * m)))
                    drop_ix = rng.choice(m, size=n_drop, replace=False)
                    keep = np.ones(m, dtype=bool)
                    keep[drop_ix] = False
                covs.append(float(np.mean(np.asarray(hits)[keep])) if keep.any() else np.nan)
            tmp = data.copy()
            tmp["coverage"] = covs
            plain = tmp.loc[tmp.personality == "plain", "coverage"].mean()
            pleasing = tmp.loc[tmp.personality == "pleasing", "coverage"].mean()
            diffs.append(100 * (pleasing - plain))
            ols = smf.ols(
                "undertriage ~ C(personality) + C(cefr) + C(recall) + C(dazed) + coverage + C(case)",
                data=tmp,
            ).fit()
            assoc.append(100 * float(ols.params[PLEASING_KEY]))
        return diffs, assoc

    for frac, nrep, label in [(0, 1, "None"), (0.10, 200, "10%"), (0.30, 200, "30%"), (0.50, 200, "50%")]:
        diffs, assoc = refit(frac, nrep)
        rows.append({
            "model": name,
            "removed": label,
            "cov_diff_median": round_half_up(float(np.median(diffs)), 2),
            "cov_diff_min": round_half_up(float(np.min(diffs)), 2),
            "cov_diff_max": round_half_up(float(np.max(diffs)), 2),
            "assoc_median": round_half_up(float(np.median(assoc)), 2),
            "assoc_min": round_half_up(float(np.min(assoc)), 2),
            "assoc_max": round_half_up(float(np.max(assoc)), 2),
        })
        print(name, label, flush=True)
    return rows


def domain_coverage(name, df):
    rows = []
    for col, levels in (
        ("personality", PERSONALITY),
        ("cefr", ["B", "A", "C"]),
        ("recall", ["high", "low"]),
        ("dazed", ["normal", "high"]),
    ):
        for level in levels:
            sub = df[df[col] == level]
            rows.append({
                "model": name,
                "domain": col,
                "level": level,
                "trials": int(len(sub)),
                "coverage_pct": round_half_up(100 * float(sub.coverage.mean()), 1),
                "turns_mean": round_half_up(float(sub.n_turns.mean()), 2),
                "patient_words_mean": round_half_up(float(sub.patient_words.mean()), 1),
                "undertriage_events": int(sub.undertriage.sum()),
            })
    return rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python analyze.py RESULTS_DIR")
    root = Path(sys.argv[1])
    out = root / "analysis_out"
    out.mkdir(parents=True, exist_ok=True)
    grouped, jsonl = discover(root)
    rates, scenarios, personas, categories = [], [], [], []
    logistic = {}
    linear = {}
    population = {}
    loaded = {}
    for model in MODELS:
        for arm, keys in (("iterative", ["case", "persona_id", "trial"]), ("control", ["case", "trial"])):
            paths = grouped[(model, arm)]
            if not paths:
                print("missing", model, arm, flush=True)
                continue
            df = load_csvs(paths, keys)
            loaded[(model, arm)] = df
            print(model, arm, int(df.undertriage.sum()), len(df), flush=True)
            rates.extend(descriptive_rows(model, arm, df))
            scenarios.extend(scenario_rows(model, arm, df))
            if arm == "iterative":
                personas.extend(persona_rows(model, df))
                categories.extend(category_rows(model, df))
                print("logistic", model, flush=True)
                logistic[model] = fit_logistic(df)
                print("linear probability", model, flush=True)
                linear[model] = fit_linear_probability(df)
                print("population logistic", model, flush=True)
                population[model] = fit_population_logistic(df)
    pd.DataFrame(rates).to_csv(out / "rates.csv", index=False)
    pd.DataFrame(scenarios).to_csv(out / "scenarios.csv", index=False)
    pd.DataFrame(personas).to_csv(out / "personas.csv", index=False)
    pd.DataFrame(categories).to_csv(out / "categories.csv", index=False)
    (out / "logistic_mixed.json").write_text(json.dumps(logistic, indent=2), encoding="utf-8")
    table1 = []
    for model in MODELS:
        if (model, "control") in loaded and (model, "iterative") in loaded:
            table1.extend(table1_rows(model, loaded[(model, "control")], loaded[(model, "iterative")]))
    pd.DataFrame(table1).to_csv(out / "table1.csv", index=False)
    (out / "linear_probability.json").write_text(json.dumps(linear, indent=2), encoding="utf-8")
    (out / "population_logistic.json").write_text(json.dumps(population, indent=2), encoding="utf-8")

    if not FINDINGS_PATH.exists():
        print("no findings.csv; coverage not scored", flush=True)
        return
    findings = load_findings()
    summaries, within, poisson, sens, domains = [], [], [], [], []
    for model in MODELS:
        if not jsonl[model] or (model, "iterative") not in loaded:
            print("no transcripts", model, flush=True)
            continue
        print("coverage", model, flush=True)
        scored, hits = score_coverage(load_records(jsonl[model]), findings)
        gold = loaded[(model, "iterative")][["case", "persona_id", "trial", "undertriage", "gold_standard", "llm_response"]]
        scored = scored.merge(gold, on=["case", "persona_id", "trial"], how="inner")
        merged = scored.merge(hits, on=["case", "persona_id", "trial"], how="inner")
        summaries.append(coverage_models(model, scored))
        within.extend(within_persona(model, scored))
        poisson.append(poisson_turns(model, scored))
        domains.extend(domain_coverage(model, scored))
        sens.extend(sensitivity(model, merged))
        scored.drop(columns=["hits"], errors="ignore").to_csv(out / f"trial_coverage_{model.replace(' ', '_')}.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "coverage_summary.csv", index=False)
    pd.DataFrame(within).to_csv(out / "within_persona_turns.csv", index=False)
    pd.DataFrame(poisson).to_csv(out / "poisson_turns.csv", index=False)
    pd.DataFrame(domains).to_csv(out / "coverage_by_domain.csv", index=False)
    pd.DataFrame(sens).to_csv(out / "coverage_sensitivity.csv", index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
