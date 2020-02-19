#!/bin/bash -e

grep -e '^# FDATA:' < "$1" | cut -c10- > "$3"
mapfile -t symbols < <(nm --defined-only "$2")

for line in "${symbols[@]}"; do
    val=$(echo $line | cut -d' ' -f1)
    symname=$(echo $line | cut -d' ' -f3)
    if [ -z "$symname" ]; then
        continue
    fi
    if [ -z "${val##*[!0-9a-fA-F]*}" ]; then
        continue
    fi
    sed -i -e "s/\#${symname}\#/$val/g" $3
done
