#!/usr/bin/env bash
# Show GPU usage on the fish machines, least-used first.
# Use this to pick the machines for launch_workers.sh.

HOSTS=(
    anchovy barracuda blowfish bonito brill bullhead char cod dorado eel
    flounder grouper halibut herring mackerel marlin perch pollock sardine
    shark sole swordfish tarpon tuna turbot wahoo
)

probe() {
    local host=$1 out used total util
    out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$host" \
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" \
        2>/dev/null | head -1)
    if [[ $out == [0-9]* ]]; then
        IFS=', ' read -r used total util <<<"$out"
        printf '%s %-12s %6s / %-6s %4s%%\n' "$used" "$host" "$used" "$total" "$util"
    else
        printf '999999 %-12s unavailable or no GPU\n' "$host"
    fi
}
export -f probe

printf '%-12s %-15s %s\n' HOST 'MEM USED (MiB)' 'GPU UTIL'
printf '%s\n' "${HOSTS[@]}" | xargs -P 26 -I{} bash -c 'probe {}' | sort -n | cut -d' ' -f2-
