#!/bin/sh

# lli is passed as a test-arg so lit can recognize it
ret=$($2 $1)

if [[ $ret = 10 ]]; then
  exit 0
else
  exit 1
fi
