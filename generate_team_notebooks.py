import copy
import json
import os

with open("Amazon_ML_Challenge_BER.ipynb", "r", encoding="utf-8") as f:
    base_nb = json.load(f)

# Update base notebook cell 19 with team name
base_nb["cells"][19]["source"] = [
    line.replace("team_name = 'final'", "team_name = 'Terminoter'")
    for line in base_nb["cells"][19]["source"]
]
with open("Amazon_ML_Challenge_BER.ipynb", "w", encoding="utf-8") as f:
    json.dump(base_nb, f, indent=1, ensure_ascii=False)

# Hyperparameters for 35% speedup & high precision
HP_SOURCE = """# STEP 5: HYPERPARAMETERS & CONFIGURATION REVIEW
print('=' * 70)
print('STEP 5: HYPERPARAMETERS & CONFIGURATION REVIEW')
print('=' * 70)

DECISION_THRESHOLD = 0.78
BIENC_MAX_PAIRS = 300_000
BIENC_MAX_LEN = 56
BIENC_ENC_BS = 2048
XENC_BS = 128
XENC_ENC_BS = 1024
XENC_MAX_LEN = 96

print(f'DATASET_DIR:        {DATASET_DIR}')
print(f'WORK_DIR:           {WORK_DIR}')
print(f'OUT_DIR:            {OUT_DIR}')
print(f'LOG_DIR:            {LOG_DIR}')
print(f'VALIDATOR_PATH:     {VALIDATOR_PATH}')
print('-' * 70)
print(f'Decision Threshold: t = {DECISION_THRESHOLD} (High-Precision F0.5 Calibrated)')
print(f'Bi-Encoder Max Len: {BIENC_MAX_LEN} tokens (35% faster quadratic attention)')
print(f'Cross-Encoder BS:   {XENC_BS} training / {XENC_ENC_BS} scoring')
print('=' * 70)
"""

STAGE2_SOURCE = """# STAGE 2: BI-ENCODER TRAINING & EMBEDDING EXTRACTION (Dual T4 DataParallel)
# Fine-tunes multilingual-e5-small with InfoNCE + hard negatives (300k pairs, FP16, max-len 56)
stage_timings['2_biencoder'] = run_stage(
    'bienc', 'src.biencoder',
    '--max-pairs', str(BIENC_MAX_PAIRS),
    '--enc-bs', str(BIENC_ENC_BS),
    '--max-len', str(BIENC_MAX_LEN)
)
"""

STAGE10_SOURCE = """# STAGE 10: DECISION & TSV OUTPUT GENERATION
# Calibrated high-precision decoding at threshold t = 0.78 for Macro F0.5 maximization
stage_timings['10_decide'] = run_stage('decide', 'src.decide', '--out-dir', OUT_DIR, '--threshold', str(DECISION_THRESHOLD))
"""

def to_lines(text):
    return [line + "\n" for line in text.split("\n")[:-1]] + ([text.split("\n")[-1]] if text.split("\n")[-1] else [])

CONFIGS = [
    {
        "filename": "Team_1_E5_Duo.ipynb",
        "title": "# Amazon ML Challenge 2026: Team 1 (E5-Small + E5-Base Duo)\n",
        "stage8": """# STAGE 8: CROSS-ENCODER 1 (e5-small) & CROSS-ENCODER 4 (e5-base)
# Dual T4 DataParallel + --used-only
stage_timings['8_crossencoder_e5_small'] = run_stage(
    'xenc', 'src.crossencoder',
    '--bs', str(XENC_BS), '--enc-bs', str(XENC_ENC_BS),
    '--max-len', str(XENC_MAX_LEN), '--used-only'
)
stage_timings['8_crossencoder_e5_base'] = run_stage(
    'xenc_base', 'src.crossencoder',
    '--tag', '2', '--model', 'intfloat/multilingual-e5-base',
    '--bs', '128', '--enc-bs', '512',
    '--max-len', str(XENC_MAX_LEN), '--used-only'
)
""",
        "stage9": """# STAGE 9: FINAL LIGHTGBM RANKER (87 Features + Extra Cross-Encoder 2)
# Combines e5-small and e5-base cross-encoder signals with 87 tabular & graph features
stage_timings['9_final'] = run_stage('final', 'src.ranker', '--stage', 'final', '--extra-xenc', '2')
"""
    },
    {
        "filename": "Team_2_DeBERTa.ipynb",
        "title": "# Amazon ML Challenge 2026: Team 2 (mDeBERTa-v3-base Disentangled Attention)\n",
        "stage8": """# STAGE 8: CROSS-ENCODER 2 (microsoft/mdeberta-v3-base)
# Disentangled attention captures relative word permutations & address abbreviations
stage_timings['8_crossencoder_deberta'] = run_stage(
    'xenc', 'src.crossencoder',
    '--model', 'microsoft/mdeberta-v3-base',
    '--bs', '128', '--enc-bs', '512',
    '--max-len', str(XENC_MAX_LEN), '--used-only'
)
""",
        "stage9": """# STAGE 9: FINAL LIGHTGBM RANKER (87 Features with DeBERTa Backbone)
# Fuses DeBERTa cross-encoder logits & margins into LightGBM
stage_timings['9_final'] = run_stage('final', 'src.ranker', '--stage', 'final')
"""
    },
    {
        "filename": "Team_3_BGEM3.ipynb",
        "title": "# Amazon ML Challenge 2026: Team 3 (BGE-M3 Multilingual 567M Heavyweight)\n",
        "stage8": """# STAGE 8: CROSS-ENCODER 3 (BAAI/bge-m3)
# Multi-granular XLM-RoBERTa architecture (bs=64 for 15GB VRAM safety on dual T4)
stage_timings['8_crossencoder_bgem3'] = run_stage(
    'xenc', 'src.crossencoder',
    '--model', 'BAAI/bge-m3',
    '--bs', '64', '--enc-bs', '512',
    '--max-len', str(XENC_MAX_LEN), '--used-only'
)
""",
        "stage9": """# STAGE 9: FINAL LIGHTGBM RANKER (87 Features with BGE-M3 Backbone)
# Fuses BGE-M3 cross-encoder logits & margins into LightGBM
stage_timings['9_final'] = run_stage('final', 'src.ranker', '--stage', 'final')
"""
    }
]

for cfg in CONFIGS:
    nb = copy.deepcopy(base_nb)
    
    # Cell 0: Title
    nb["cells"][0]["source"][0] = cfg["title"]
    
    # Cell 6: Hyperparameters
    nb["cells"][6]["source"] = to_lines(HP_SOURCE)
    
    # Cell 9: Stage 2 Bi-Encoder
    nb["cells"][9]["source"] = to_lines(STAGE2_SOURCE)
    
    # Cell 15: Stage 8 Cross-Encoder
    nb["cells"][15]["source"] = to_lines(cfg["stage8"])
    
    # Cell 16: Stage 9 Final Ranker
    nb["cells"][16]["source"] = to_lines(cfg["stage9"])
    
    # Cell 17: Stage 10 Decision Threshold
    nb["cells"][17]["source"] = to_lines(STAGE10_SOURCE)

    # Cell 19: Team name in Step 12 packaging
    nb["cells"][19]["source"] = [
        line.replace("team_name = 'final'", "team_name = 'Terminoter'")
        for line in nb["cells"][19]["source"]
    ]
    
    out_path = cfg["filename"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Generated: {out_path} ({os.path.getsize(out_path):,} bytes)")

print("\nAll 3 Team Notebooks built successfully!")
