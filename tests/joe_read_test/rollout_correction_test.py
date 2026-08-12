from verl.trainer.config.algorithm import RolloutCorrectionConfig

# === Decoupled PPO mode (3 policies: π_rollout, π_old, π_θ) ===
# IS weights correct for gap between π_old and π_rollout
config = RolloutCorrectionConfig.decoupled_token_is()           # Token-TIS
config = RolloutCorrectionConfig.decoupled_seq_is()             # Seq-TIS
config = RolloutCorrectionConfig.decoupled_seq_is_rs()          # Seq-MIS
config = RolloutCorrectionConfig.decoupled_geo_rs()             # Geo-RS (ratio mode)
config = RolloutCorrectionConfig.decoupled_geo_rs_token_tis()   # Geo-RS + Token-TIS

# === K3 KL Estimator presets (more stable for small KL) ===
config = RolloutCorrectionConfig.decoupled_k3_rs()              # K3-RS only
config = RolloutCorrectionConfig.decoupled_k3_rs_token_tis()    # K3-RS + Token-TIS

# === Bypass PPO mode (2 policies: π_rollout = π_old, π_θ) - fast ===
# PPO ratio handles IS, so no explicit IS weights needed
config = RolloutCorrectionConfig.bypass_ppo_clip()              # PPO-clip only
config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs()       # PPO-clip + Geo-RS
config = RolloutCorrectionConfig.bypass_ppo_clip_k3_rs()        # PPO-clip + K3-RS

# === Bypass PG mode (2 policies, no PPO clipping) - fast ===
# IS weights computed on-the-fly as π_θ / π_rollout
config = RolloutCorrectionConfig.bypass_pg_is()                 # Seq-TIS + PG
config = RolloutCorrectionConfig.bypass_pg_geo_rs()             # Geo-RS + PG
config = RolloutCorrectionConfig.bypass_pg_geo_rs_token_tis()   # Geo-RS + Token-TIS + PG

# === Other ===
config = RolloutCorrectionConfig.disabled()             # Metrics only (no correction)