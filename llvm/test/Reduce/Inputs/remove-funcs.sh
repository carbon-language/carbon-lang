#!/bin/sh

matches=$(cat $1 | grep "@interesting" | wc -l)

if [[ $matches > 0 ]]; then
  exit 0
else
  exit 1
fi
