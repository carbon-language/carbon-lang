#!/bin/bash

ulimit -s 8192;
set -e # fail on any error

ROOTDIR=$(dirname $0)/..

# Assuming clang is in path.
CC=clang
CXX=clang++

# TODO: add testing for all of -O0...-O3
CFLAGS="-fthread-sanitizer -fPIE -O1 -g -fno-builtin -Wall -Werror=return-type"
LDFLAGS="-pie -lpthread -ldl $ROOTDIR/rtl/libtsan.a"
if [ "$LLDB" != "" ]; then
  LDFLAGS+=" -L$LLDB -llldb"
fi

test_file() {
  SRC=$1
  COMPILER=$2
  echo ----- TESTING $(basename $1)
  OBJ=$SRC.o
  EXE=$SRC.exe
  $COMPILER $SRC $CFLAGS -c -o $OBJ
  # Link with CXX, because lldb and suppressions require C++.
  $CXX $OBJ $LDFLAGS -o $EXE
  RES=$(LD_LIBRARY_PATH=$LLDB TSAN_OPTIONS="atexit_sleep_ms=0" $EXE 2>&1 || true)
  if [ "$3" != "" ]; then
    printf "%s\n" "$RES"
  fi
  printf "%s\n" "$RES" | FileCheck $SRC
  if [ "$3" == "" ]; then
    rm -f $EXE $OBJ
  fi
}

if [ "$1" == "" ]; then
  for c in $ROOTDIR/output_tests/*.{c,cc}; do
    if [[ $c == */failing_* ]]; then
      echo SKIPPING FAILING TEST $c
      continue
    fi
    COMPILER=$CXX
    case $c in
      *.c) COMPILER=$CC
    esac
    test_file $c $COMPILER &
  done
  wait
else
  test_file $ROOTDIR/output_tests/$1 $CXX "DUMP"
fi
