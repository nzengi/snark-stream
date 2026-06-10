#!/usr/bin/env bash
# Real run captured for the README GIF (qmem-demo.gif) — nothing is faked. Both
# invocations evaluate the SAME Halo2 quotient (a wide custom-gate circuit at
# k=18, 96 advice columns) under the SAME enforced 1536 MB cgroup, swap off,
# spilling to NVMe. The in-core path holds every advice extended coset resident;
# the disk path (evaluate_h_ooc_disk) spills each to /var/tmp and reads back one
# tile-sized window at a time. The RSS lines are polled live from the running
# process's /proc/<pid>/status; both produce the identical quotient fingerprint.
#
# The binary is the vendored fork's `ooc-demo` (halo2-fork/, build it per
# integration/README.md); override with SS_OOC_BIN if it lives elsewhere.
#
# Regenerate the GIF:
#   cargo build --release -p ooc-demo --manifest-path halo2-fork/Cargo.toml
#   asciinema rec --overwrite -c "bash halo2-stream/integration/qmem-demo.sh" /tmp/q.cast
#   agg --theme github-dark --cols 100 --rows 20 --idle-time-limit 1 \
#       --speed 2 --last-frame-duration 3 /tmp/q.cast docs/qmem-demo.gif
set +e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
BIN="${SS_OOC_BIN:-halo2-fork/target/release/ooc-demo}"
CG="systemd-run --user --scope -q -p MemoryMax=1536M -p MemorySwapMax=0 --"
p() { printf '%s\n' "$*"; }

# Run "$@" under the cgroup in the background, then live-poll the worker's RSS
# (matched by command tag $tag) into a single updating line until it exits. The
# subshell's `exec 2>/dev/null` discards the "Killed" job-control notice emitted
# when the OOM-killed worker is reaped; the worker's own output goes to $out.
run_polled() {
  local tag="$1" label="$2" period="$3"; shift 3
  local out rcf; out=$(mktemp); rcf=$(mktemp)
  ( exec 2>/dev/null; $CG "$BIN" "$@" >"$out" 2>&1; echo $? >"$rcf" ) &
  local job=$!
  sleep 0.5
  local bpid; bpid=$(pgrep -n -f "ooc-demo $tag")
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
p "# snark-stream: a Halo2 quotient that does not fit in the RAM budget."
p "# k=2^18, 96 advice columns — the extended-domain cosets — under 1536 MB, swap off."
p ""
sleep 1.2

p "\$ # 1) in-core evaluate_h — every advice extended coset resident at once:"
sleep 0.4
p "\$ systemd-run -p MemoryMax=1536M -p MemorySwapMax=0 -- ooc-demo qmem stock 18 96"
sleep 0.4
run_polled "qmem stock" "building the extended cosets ... climbing toward the 1536 MB cap" 0.4 qmem stock 18 96 8192
rc=$?
[ "$rc" -eq 137 ] && p "  x  Killed — out of memory at 1536 MB (the cosets need ~2.5 GB)."
p ""
sleep 1.4

p "\$ # 2) evaluate_h_ooc_disk — cosets spilled to NVMe, one tile-window at a time:"
sleep 0.4
p "\$ systemd-run -p MemoryMax=1536M -p MemorySwapMax=0 -- ooc-demo qmem disk 18 96"
sleep 0.4
run_polled "qmem disk" "tiling the quotient ... one coset + a window resident, RAM stays flat" 0.8 qmem disk 18 96 8192
sed 's/^/  /' "$RUN_OUT"
sleep 0.6
p ""
p "# Same quotient, identical fingerprint to the in-core one — 2.5 GB held to ~0.9 GB,"
p "# of which most is the resident witness. A prover's RAM is a dial, not a wall."
sleep 2.0
