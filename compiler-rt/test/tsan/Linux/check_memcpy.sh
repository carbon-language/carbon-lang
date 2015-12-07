#!/bin/bash

# Script that ensures that TSan runtime does not contain compiler-emitted
# memcpy and memset calls.

set -eu

if [[ "$#" != 1 ]]; then
  echo "Usage: $0 /path/to/binary/built/with/tsan"
  exit 1
fi

EXE=$1

NCALL=$(objdump -d $EXE | egrep "callq .*<__interceptor_mem(cpy|set)>" | wc -l)
if [ "$NCALL" != "0" ]; then
  echo FAIL: found $NCALL memcpy/memset calls
  exit 1
fi

# tail calls
NCALL=$(objdump -d $EXE | egrep "jmpq .*<__interceptor_mem(cpy|set)>" | wc -l)
if [ "$NCALL" != "0" ]; then
  echo FAIL: found $NCALL memcpy/memset calls
  exit 1
fi
