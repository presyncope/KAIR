#!/usr/bin/env bash
# run_train.sh
# - nohup + systemd-run --scope 로 RAM 40GB 제한
# - PID/로그 관리
# - 추가 인자 전달 가능: ./run_train.sh --epochs 100 ...

set -euo pipefail

### 기본 설정 (요청 반영)
PY=${PY:-python}                               # 기본 python
ENTRY=${ENTRY:-main_train_rebotnet.py}         # 엔트리 스크립트
MEM_LIMIT=${MEM_LIMIT:-52G}                    # RAM 상한
SWAP_LIMIT=${SWAP_LIMIT:-0}                    # 0이면 스왑 금지
WORKDIR=${WORKDIR:-"$(pwd)"}                   # 기본 작업 디렉토리

### 내부 설정
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-"${WORKDIR}/logs"}
PID_DIR=${PID_DIR:-"${WORKDIR}/run"}
UNIT_NAME=${UNIT_NAME:-"train_${STAMP}"}

# echo "[INFO] remounting /dev/shm with size=50G..."
# sudo mount -o remount,size=50G /dev/shm

# echo "[INFO] current /dev/shm status:"
# df -h /dev/shm

mkdir -p "$LOG_DIR" "$PID_DIR"

LOG_FILE="${LOG_DIR}/${UNIT_NAME}.log"
PID_FILE="${PID_DIR}/${UNIT_NAME}.pid"

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "[ERROR] systemd-run 이 필요합니다." >&2
  exit 1
fi

echo "[INFO] Starting ${ENTRY} with MemoryMax=${MEM_LIMIT}, SwapMax=${SWAP_LIMIT}"
echo "[INFO] Logs: ${LOG_FILE}"
echo "[INFO] PID file: ${PID_FILE}"

# 안쪽 bash에서 사용할 변수들을 환경변수로 전달
export WORKDIR PID_FILE PY ENTRY

NOHUP_BASE=( nohup systemd-run --user --scope -p "MemoryMax=${MEM_LIMIT}" )
if [[ -n "${SWAP_LIMIT}" ]]; then
  NOHUP_BASE+=( -p "MemorySwapMax=${SWAP_LIMIT}" )
fi

# 핵심: bash -c '... "$@"' bash "$@" 패턴
# - 인자가 없으면 아무것도 전달되지 않음(빈 문자열 방지)
# - $$로 실제 실행 PID 기록
"${NOHUP_BASE[@]}" \
  bash -c 'cd "$WORKDIR" && echo $$ > "$PID_FILE" && exec "$PY" "$ENTRY" "$@"' \
  bash "$@" >> "${LOG_FILE}" 2>&1 &

sleep 0.3
if [[ -s "${PID_FILE}" ]]; then
  PY_PID=$(cat "${PID_FILE}")
  echo "[INFO] Launched. Python PID: ${PY_PID}"
  echo "[INFO] Tail logs: tail -f \"${LOG_FILE}\""
else
  echo "[WARN] PID 파일이 비어있습니다. 실행 상태는 로그로 확인하세요: ${LOG_FILE}"
fi
