#!/bin/bash
# Compile Antonina Nazarova's CapSelect 2021 source locally as a "ground
# truth" binary. The compiled binary's behavior matches the source-as-
# documented (e.g. with the ip2_1 cap-clearance constraint added Sep 2021),
# whereas the prebuilt Linux binary at
#   /mnt/katritch_lab/Antonina/CapSelection/Example_project_input/CapSelection
# was compiled in June 2021 and predates that constraint.
#
# Usage:
#   bash build_local_binary.sh
#
# Produces: ./test_data/CapSelect_local
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p test_data
g++ -O2 -std=c++17 -pthread -w sources/CapSelect_2021.cpp -o test_data/CapSelect_local
echo "wrote test_data/CapSelect_local"
file test_data/CapSelect_local
