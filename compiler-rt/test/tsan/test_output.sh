#!/bin/bash

ulimit -s 8192
set -e # fail on any error

HERE=$(dirname $0)
TSAN_DIR=$(dirname $0)/../../lib/tsan

# Assume clang and clang++ are in path.
: ${CC:=clang}
: ${CXX:=clang++}
: ${FILECHECK:=FileCheck}

# TODO: add testing for all of -O0...-O3
CFLAGS="-fsanitize=thread -fPIE -O1 -g -Wall"
LDFLAGS="-pie -pthread -ldl -lrt -lm -Wl,--whole-archive $TSAN_DIR/rtl/libtsan.a -Wl,--no-whole-archive"

test_file() {
  SRC=$1
  COMPILER=$2
  echo ----- TESTING $(basename $1)
  OBJ=$SRC.o
  EXE=$SRC.exe
  $COMPILER $SRC $CFLAGS -c -o $OBJ
  $COMPILER $OBJ $LDFLAGS -o $EXE
  RES=$($EXE 2>&1 || true)
  printf "%s\n" "$RES" | $FILECHECK $SRC
  if [ "$3" == "" ]; then
    rm -f $EXE $OBJ
  fi
}

if [ "$1" == "" ]; then
  for c in $HERE/*.{c,cc}; do
    if [[ $c == */failing_* ]]; then
      echo SKIPPING FAILING TEST $c
      continue
    fi
    if [[ $c == */load_shared_lib.cc ]]; then
      echo TEST $c is not supported
      continue
    fi
    if [[ $c == */*blacklist*.cc ]]; then
      echo TEST $c is not supported
      continue
    fi
    if [ "`grep "TSAN_OPTIONS" $c`" ]; then
      echo SKIPPING $c -- requires TSAN_OPTIONS
      continue
    fi
    COMPILER=$CXX
    case $c in
      *.c) COMPILER=$CC
    esac
    test_file $c $COMPILER &
  done
  for job in `jobs -p`; do
    wait $job || exit 1
  done
else
  test_file $HERE/$1 $CXX "DUMP"
fi
