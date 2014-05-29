#!/bin/bash

# Ensure that tsan runtime does not contain compiler-emitted memcpy and memset calls.

set -eu

ROOTDIR=$(dirname $0)
TEST_DIR=$ROOTDIR/../../test/tsan

: ${CXX:=clang++}
CFLAGS="-fsanitize=thread -fPIE -O1 -g"
LDFLAGS="-pie -lpthread -ldl -lrt -lm -Wl,--whole-archive $ROOTDIR/rtl/libtsan.a -Wl,--no-whole-archive"

SRC=$TEST_DIR/simple_race.cc
OBJ=$SRC.o
EXE=$SRC.exe
$CXX $SRC $CFLAGS -c -o $OBJ
$CXX $OBJ $LDFLAGS -o $EXE

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
