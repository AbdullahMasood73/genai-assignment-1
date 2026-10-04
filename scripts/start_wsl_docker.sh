#!/usr/bin/env bash
# Optional isolated engine for the existing Ubuntu/WSL deployment check.
set -euo pipefail
if [ "$(id -u)" != 0 ]; then
  echo 'Run this helper as root inside your existing WSL Ubuntu distro.' >&2
  exit 1
fi
prefix=/tmp/restoration-lab-docker
runtime_bin="${prefix}-tools"
mkdir -p "$runtime_bin"
for executable in iptables ip6tables; do
  target="/usr/sbin/${executable}-legacy"
  if [ ! -x "$target" ]; then
    echo "Missing existing compatibility executable: $target" >&2
    exit 1
  fi
  if [ -e "$runtime_bin/$executable" ] || [ -L "$runtime_bin/$executable" ]; then
    if [ "$(readlink "$runtime_bin/$executable")" != "$target" ]; then
      echo 'Unexpected file in temporary helper directory; stopping.' >&2
      exit 1
    fi
  else
    ln -s "$target" "$runtime_bin/$executable"
  fi
done
printf '{}\n' > "$runtime_bin/daemon.json"
exec env PATH="$runtime_bin:/usr/sbin:/usr/bin:/sbin:/bin" dockerd \
  --config-file "$runtime_bin/daemon.json" \
  --data-root "${prefix}-data" --exec-root "${prefix}-exec" \
  --pidfile "${prefix}.pid" --host "unix://${prefix}.sock"
