#!/bin/bash
hrule() {
  local n="${1:-${COLUMNS:-80}}"
  printf '%*s\n' "$n" '' | tr ' ' '-'
}
info() {
  hrule
  printf '%s\n' "$*"
  hrule
}
set -euo pipefail
info "STEP 1: Creating sentifm conda environment"
conda create -n sentifm python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sentifm
info "STEP 2: Installing dependencies"
uv pip install osfclient spacy tqdm
python -m spacy download en_core_web_sm
info "STEP 3: Downloading SentiFM dataset"
osf -p enu2k clone .
info "STEP 4: Uncompressing raw documents"
mkdir -p bratdata
tar -xzf osfstorage/bratannotationfiles.tar.gz -C bratdata
info "STEP 5: Splitting documents into sentences"
mkdir -p sentences
python split.py \
  --input_dir bratdata/bratannotationfiles \
  --output_tsv sentences/splitted.tsv \
  --extra_split
info "STEP 6: Removing noisy sentences"
python prune.py \
  sentences/splitted.tsv \
  sentences/pruned.tsv
info "STEP 7: Converting to JSONL format"
python convert.py \
  sentences/pruned.tsv \
  sentences/pruned.jsonl
info "SUCCESS: Built sentences/pruned.jsonl with $(cat sentences/pruned.jsonl | wc -l) sentences"
