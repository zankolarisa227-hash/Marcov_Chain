from pathlib import Path
import pandas as pd

def load(dir_):
    files = sorted(Path(dir_).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

panel = load("data/processed/panel_synth")
feat = load("data/processed/features_synth")
feat = feat.sort_values(["client_id", "month"])
feat["next_state"] = feat.groupby("client_id")["state"].shift(-1)
feat = feat.dropna(subset=["next_state"])

print("panel", panel.shape, "clients", panel.client_id.nunique())
print("features", feat.shape)
print("\nstate share\n", panel.state.value_counts(normalize=True).sort_index())
print("\nmonths per client\n", panel.groupby("client_id").size().describe())
print("\ndefault rate by state")
print(feat.groupby("state")["next_state"].apply(lambda s: (s == 2).mean()))
print("\nutilization by next_state")
print(feat.groupby("next_state")["utilization"].describe())