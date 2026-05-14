#!/bin/bash
#SBATCH -J unet_reg_eval
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/demo_classification.py" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../demo_classification.py" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: cannot locate repository root from submit directory: ${SUBMIT_DIR}"
    echo "Submit this job from the repository root or from the run directory."
    exit 1
fi

RUN_DIR="${REPO_ROOT}/run"
PARENT_DATA_DIR="${REPO_ROOT}/../data"
export DATA_ROOT="${PARENT_DATA_DIR}"

DATASET="amplitude"
LOSS="wdice"
NET_TYPE="unet"
VIT_TYPE="small"
CHECKPOINT_NAME="unfrozen"

SLURM_LOG_DIR="${RUN_DIR}/slurm_logs"
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/unet_reg_eval"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoint/${DATASET}/${LOSS}/${NET_TYPE}"
MODEL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}_${VIT_TYPE}_maxiou_valid.pth"
TRAIN_LOG_DIR="${REPO_ROOT}/log/${DATASET}/${LOSS}/${NET_TYPE}"
PNG_DIR="${REPO_ROOT}/png/${DATASET}/${NET_TYPE}/${LOSS}/${CHECKPOINT_NAME}_${VIT_TYPE}"

mkdir -p "${SCRIPT_LOG_DIR}" "${TRAIN_LOG_DIR}" "${PNG_DIR}"
exec > "${JOB_OUT}" 2> "${JOB_ERR}"

module load nvidia/cuda

BOOTSTRAP_PYTHON=""
for PY_CANDIDATE in python3.11 python3.10 python3.9 python3.8 python3 python; do
    if command -v "${PY_CANDIDATE}" >/dev/null 2>&1; then
        PY_CANDIDATE_PATH="$(command -v "${PY_CANDIDATE}")"
        if "${PY_CANDIDATE_PATH}" -c 'import sys; raise SystemExit(0 if (3, 8) <= sys.version_info[:2] < (3, 12) else 1)' >/dev/null 2>&1; then
            BOOTSTRAP_PYTHON="${PY_CANDIDATE_PATH}"
            break
        fi
    fi
done

if [ -z "${BOOTSTRAP_PYTHON}" ]; then
    echo "ERROR: compatible Python interpreter not found."
    echo "This project requirements need Python 3.8-3.11; Python 3.12 breaks numpy==1.23.4."
    echo "Load a compatible Python module, for example python/3.10 or python/3.11 if available."
    echo "CUDA module is loaded according to instruction_cluster.doc, but Python is not listed there."
    exit 1
fi

VENV_DIR="${REPO_ROOT}/.venv"

echo "job_id=${SLURM_JOBID:-manual}"
echo "node=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "dataset=${PARENT_DATA_DIR}/${DATASET}"
echo "model_path=${MODEL_PATH}"
echo "eval_log=${TRAIN_LOG_DIR}/${CHECKPOINT_NAME}_${VIT_TYPE}.txt"
echo "png=${PNG_DIR}"
nvidia-smi

if [ -x "${VENV_DIR}/bin/python" ]; then
    if ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(0 if (3, 8) <= sys.version_info[:2] < (3, 12) else 1)' >/dev/null 2>&1; then
        echo "ERROR: existing ${VENV_DIR} uses an incompatible Python."
        echo "Remove it and rerun: rm -rf ${VENV_DIR}"
        echo "This project requirements need Python 3.8-3.11; Python 3.12 breaks numpy==1.23.4."
        exit 1
    fi
else
    echo "Creating virtual environment: ${VENV_DIR}"
    "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
PYTHON_BIN="${VENV_DIR}/bin/python"

REQ_HASH="$(sha256sum "${REPO_ROOT}/requirements.txt" | awk '{print $1}')"
REQ_MARKER="${VENV_DIR}/.requirements-${REQ_HASH}.installed"
REQ_CLEAN="${VENV_DIR}/requirements.cluster.txt"

if [ ! -f "${REQ_MARKER}" ]; then
    echo "Installing Python dependencies into ${VENV_DIR}"
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    grep -v '^Pillow==9\.2\.0$' "${REPO_ROOT}/requirements.txt" > "${REQ_CLEAN}"
    "${PYTHON_BIN}" -m pip install -r "${REQ_CLEAN}"
    rm -f "${VENV_DIR}"/.requirements-*.installed
    touch "${REQ_MARKER}"
else
    echo "Python dependencies already installed for current requirements.txt"
fi

echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version

if [ ! -d "${PARENT_DATA_DIR}/${DATASET}" ]; then
    echo "ERROR: dataset not found."
    echo "Expected ${PARENT_DATA_DIR}/${DATASET}"
    echo "The evaluation code reads this path from DATA_ROOT=${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: trained checkpoint not found."
    echo "Expected ${MODEL_PATH}"
    echo "The evaluator loads exactly this file for -net ${NET_TYPE} -d ${DATASET} -loss ${LOSS} -cp ${CHECKPOINT_NAME} -v ${VIT_TYPE}"
    exit 1
fi
echo "valid_files=$(find "${PARENT_DATA_DIR}/${DATASET}/valid/input" -maxdepth 1 -type f -name '*.dat' | wc -l)"

cd "${RUN_DIR}"

echo "---------------Evaluate UNet facies/amplitude----------------"
"${PYTHON_BIN}" ../evaluate_classification.py \
    -net "${NET_TYPE}" \
    -d "${DATASET}" \
    -dn "cuda" \
    -v "${VIT_TYPE}" \
    -loss "${LOSS}" \
    -cp "${CHECKPOINT_NAME}"

echo "Done"
