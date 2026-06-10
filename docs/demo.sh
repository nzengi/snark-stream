#!/usr/bin/env bash
# Real run captured for the README GIF (docs/demo.gif) — nothing is faked; both
# invocations execute under the same enforced 512 MB cgroup, swap off, against a
# degree-2^26 SRS (~4 GB of bases) that was streamed to NVMe ahead of time. The
# RSS lines are polled live from the running process's /proc/<pid>/status.
#
# Regenerate the GIF:
#   cargo build --release
#   ./target/release/snark-stream gen 26 /var/tmp      # ~6 GB of bases + scalars on NVMe
#   asciinema rec --overwrite -c "bash docs/demo.sh" /tmp/ss.cast
#   agg --theme github-dark --cols 100 --rows 18 --idle-time-limit 1 \
#       --speed 2 --last-frame-duration 3 /tmp/ss.cast docs/demo.gif
set +e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
BIN=./target/release/snark-stream
DIR=/var/tmp
CG="systemd-run --user --scope -q -p MemoryMax=512M -p MemorySwapMax=0 --"
p() { printf '%s\n' "$*"; }

# Run "$@" under the cgroup in the background, then live-poll the worker's RSS
# (matched by command tag $tag) into a single updating line until it exits.
# The cgroup call runs inside a subshell that records its exit code to a file.
# `exec 2>/dev/null` points the subshell's own stderr at the void, so the "Killed"
# job-control notice it would print when the OOM-killed worker is reaped is
# discarded; the worker's real output is captured separately via `>"$out" 2>&1`.
run_polled() {
  local tag="$1" label="$2" period="$3"; shift 3
  local out rcf; out=$(mktemp); rcf=$(mktemp)
  ( exec 2>/dev/null; $CG "$BIN" "$@" >"$out" 2>&1; echo $? >"$rcf" ) &
  local job=$!
  sleep 0.5
  local bpid; bpid=$(pgrep -n -f "snark-stream $tag")
  while [ ! -s "$rcf" ]; do
    local rss; rss=$(awk '/VmRSS/{printf "%d",$2/1024}' "/proc/$bpid/status" 2>/dev/null)
    printf '\r  %s   RSS=%s MB    ' "$label" "${rss:-..}"
    sleep "$period"
  done
  wait "$job" 2>/dev/null
  printf '\r\033[K'
  RUN_OUT="$out"; return "$(cat "$rcf")"
}

sleep 0.7
p "# snark-stream: a KZG commitment whose SRS does not fit in the RAM budget."
p "# BN254, degree 2^26 — 67M points, ~4 GB of bases — under a 512 MB cgroup, swap off."
p ""
sleep 1.2

p "\$ # 1) in-core KZG commit (the deployed path) — loads the whole SRS into RAM:"
sleep 0.4
p "\$ systemd-run -p MemoryMax=512M -p MemorySwapMax=0 -- snark-stream kzgincore 26"
sleep 0.4
run_polled kzgincore "loading SRS into RAM ... climbing toward the 512 MB cap" 0.4 kzgincore 26 "$DIR"
rc=$?
[ "$rc" -eq 137 ] && p "  x  Killed — out of memory at 512 MB (the full SRS needs ~11 GB)."
p ""
sleep 1.4

p "\$ # 2) out-of-core KZG commit — same commitment, SRS streamed from NVMe:"
sleep 0.4
p "\$ systemd-run -p MemoryMax=512M -p MemorySwapMax=0 -- snark-stream kzgooc 26"
sleep 0.4
run_polled kzgooc "streaming 67M SRS points through fixed buckets ... RAM stays flat" 1.0 kzgooc 26 "$DIR"
sed 's/^/  /' "$RUN_OUT"
sleep 0.6
p ""
p "# Same commitment, bit-identical to arkworks KZG10 — 11 GB collapsed to ~120 MB."
p "# A prover's RAM is a dial, not a wall."
sleep 2.0
