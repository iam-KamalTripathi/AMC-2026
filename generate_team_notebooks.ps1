$ProgressPreference = 'SilentlyContinue'

$baseNbPath = "Amazon_ML_Challenge_BER.ipynb"
$baseJson = [System.IO.File]::ReadAllText((Resolve-Path $baseNbPath), [System.Text.Encoding]::UTF8)

function Make-Notebook($nbJson, $title, $stage8Py, $stage9Py, $destPath) {
    $nb = $nbJson | ConvertFrom-Json
    
    # Update Cell 0 Title
    $nb.cells[0].source[0] = "# $title`n"
    
    # Update Cell 6 (Hyperparameters)
    $hpPy = @"
# STEP 5: HYPERPARAMETERS & CONFIGURATION REVIEW
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
"@
    $hpLines = ($hpPy -split "`r?`n") | ForEach-Object { $_ + "`n" }
    $nb.cells[6].source = $hpLines

    # Update Cell 9 (Stage 2 Bi-Encoder to use BIENC_MAX_LEN and BIENC_MAX_PAIRS)
    $st2Py = @"
# STAGE 2: BI-ENCODER TRAINING & EMBEDDING EXTRACTION (Dual T4 DataParallel)
# Fine-tunes multilingual-e5-small with InfoNCE + hard negatives (300k pairs, FP16, max-len 56)
stage_timings['2_biencoder'] = run_stage(
    'bienc', 'src.biencoder',
    '--max-pairs', str(BIENC_MAX_PAIRS),
    '--enc-bs', str(BIENC_ENC_BS),
    '--max-len', str(BIENC_MAX_LEN)
)
"@
    $st2Lines = ($st2Py -split "`r?`n") | ForEach-Object { $_ + "`n" }
    $nb.cells[9].source = $st2Lines

    # Update Cell 15 (Stage 8 Cross-Encoder)
    $st8Lines = ($stage8Py -split "`r?`n") | ForEach-Object { $_ + "`n" }
    $nb.cells[15].source = $st8Lines

    # Update Cell 16 (Stage 9 Final Ranker)
    $st9Lines = ($stage9Py -split "`r?`n") | ForEach-Object { $_ + "`n" }
    $nb.cells[16].source = $st9Lines

    # Update Cell 17 (Stage 10 Decision Threshold)
    $st10Py = @"
# STAGE 10: DECISION & TSV OUTPUT GENERATION
# Calibrated high-precision decoding at threshold t = 0.78 for Macro F0.5 maximization
stage_timings['10_decide'] = run_stage('decide', 'src.decide', '--out-dir', OUT_DIR, '--threshold', str(DECISION_THRESHOLD))
"@
    $st10Lines = ($st10Py -split "`r?`n") | ForEach-Object { $_ + "`n" }
    $nb.cells[17].source = $st10Lines

    $updated = $nb | ConvertTo-Json -Depth 32
    [System.IO.File]::WriteAllText($destPath, $updated, [System.Text.Encoding]::UTF8)
    Write-Output "Generated: $destPath"
}

# 1. Notebook 1: Teammate 1 (e5-small + e5-base duo)
$nb1_st8 = @"
# STAGE 8: CROSS-ENCODER 1 (e5-small) & CROSS-ENCODER 4 (e5-base)
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
"@
$nb1_st9 = @"
# STAGE 9: FINAL LIGHTGBM RANKER (87 Features + Extra Cross-Encoder 2)
stage_timings['9_ranker'] = run_stage('ranker', 'src.ranker', '--mode', 'final', '--extra-xenc', '2')
"@
Make-Notebook $baseJson "Amazon ML BER - Team 1 (E5-Small + E5-Base Duo)" $nb1_st8 $nb1_st9 "Team_1_E5_Duo.ipynb"

# 2. Notebook 2: Teammate 2 (DeBERTa Disentangled Attention)
$nb2_st8 = @"
# STAGE 8: CROSS-ENCODER 2 (microsoft/mdeberta-v3-base)
# Disentangled attention for address order swaps & spelling discrepancies
stage_timings['8_crossencoder_deberta'] = run_stage(
    'xenc', 'src.crossencoder',
    '--model', 'microsoft/mdeberta-v3-base',
    '--bs', '128', '--enc-bs', '512',
    '--max-len', str(XENC_MAX_LEN), '--used-only'
)
"@
$nb2_st9 = @"
# STAGE 9: FINAL LIGHTGBM RANKER (87 Features with DeBERTa Backbone)
stage_timings['9_ranker'] = run_stage('ranker', 'src.ranker', '--mode', 'final')
"@
Make-Notebook $baseJson "Amazon ML BER - Team 2 (DeBERTa Syntactic Model)" $nb2_st8 $nb2_st9 "Team_2_DeBERTa.ipynb"

# 3. Notebook 3: Teammate 3 (BGE-M3 Multilingual Heavyweight)
$nb3_st8 = @"
# STAGE 8: CROSS-ENCODER 3 (BAAI/bge-m3)
# Multi-granular XLM-RoBERTa for transliterations & script variations (bs=64 for 15GB VRAM safety)
stage_timings['8_crossencoder_bgem3'] = run_stage(
    'xenc', 'src.crossencoder',
    '--model', 'BAAI/bge-m3',
    '--bs', '64', '--enc-bs', '512',
    '--max-len', str(XENC_MAX_LEN), '--used-only'
)
"@
$nb3_st9 = @"
# STAGE 9: FINAL LIGHTGBM RANKER (87 Features with BGE-M3 Backbone)
stage_timings['9_ranker'] = run_stage('ranker', 'src.ranker', '--mode', 'final')
"@
Make-Notebook $baseJson "Amazon ML BER - Team 3 (BGE-M3 Multilingual Champion)" $nb3_st8 $nb3_st9 "Team_3_BGEM3.ipynb"

Write-Output "All 3 team notebooks successfully built!"
