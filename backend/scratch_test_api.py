import urllib.request
import json

base = "http://127.0.0.1:8001/api"

def check(endpoint):
    url = f"{base}/{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        size = len(data) if isinstance(data, list) else len(data.keys())
        print(f"SUCCESS [{endpoint}]: response type {type(data).__name__}, count/keys={size}")
        return data

print("Testing Research and Intelligence API endpoints...")

# 1. Experiments
exps = check("research/experiments")
print(f"  First experiment: {exps[0]['experiment_id']} - {exps[0]['name']}, Brier={exps[0]['brier_score']}")
print(f"  Candidate ensemble: {exps[-1]['experiment_id']} - {exps[-1]['name']}, Status={exps[-1]['status']}, Sharpe={exps[-1]['net_sharpe']}")

# 2. Graveyard
gy = check("research/graveyard")
print(f"  First graveyard record: {gy[0]['experiment_id']} - {gy[0]['flaw_type']}")

# 3. HMM
hmm = check("research/hmm/ARB/USDT")
print(f"  HMM dominant state: {hmm['state_posteriors']['dominant_state']}, Recovery={hmm['state_posteriors']['recovery']}%, Modal window: {hmm['modal_window']}")

# 4. Large-Move Surface (BR-001.1)
lm = check("research/large-move/ARB/USDT")
print(f"  Expected MFE 30D: +{lm['expected_mfe_30d_pct']}%, Asymmetry Ratio: {lm['asymmetry_ratio']}x")

# 5. Entry Strategies (BR-001.2)
es = check("research/entry-strategies/ARB/USDT")
print(f"  Entry strategies count: {len(es)}, E4 recommendation: {es[-1]['recommendation']}")

# 6. False-Recovery (BR-001.3)
fr = check("research/false-recovery/ARB/USDT")
print(f"  False-recovery risk level: {fr['risk_level']}, Failure prob: {fr['p_recovery_failure_pct']}%")

# 7. Universe Ranking (BR-001.4)
rk = check("research/ranking?limit=5")
print(f"  Universe ranked count: {len(rk)}, Top asset: {rk[0]['symbol']} (Score {rk[0]['opportunity_score']})")

# 8. Asset Intelligence
intel = check("intelligence/ARB/USDT")
print(f"  Intelligence decision: {intel['decision']}, Bear Exhaustion={intel['bear_exhaustion']['score']}/100, Primary Regime={intel['regime_distribution']['primary_regime']}")

print("\nALL HTTP REST API ENDPOINTS VERIFIED 100% OPERATIONAL!")
