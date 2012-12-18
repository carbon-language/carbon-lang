#!/bin/bash

ulimit -s 8192
set -e # fail on any error

ROOTDIR=$(dirname $0)/..

# Assuming clang is in path.
CC=clang
CXX=clang++

# TODO: add testing for all of -O0...-O3
CFLAGS="-fsanitize=thread -fPIE -O1 -g -fno-builtin -Wall"
LDFLAGS="-pie -lpthread -ldl $ROOTDIR/rtl/libtsan.a"

test_file() {
  SRC=$1
  COMPILER=$2
  echo ----- TESTING $(basename $1)
  OBJ=$SRC.o
  EXE=$SRC.exe
  $COMPILER $SRC $CFLAGS -c -o $OBJ
  $COMPILER $OBJ $LDFLAGS -o $EXE
  RES=$($EXE 2>&1 || true)
  if [ "$3" != "" ]; then
    printf "%s\n" "$RES"
  fi
  printf "%s\n" "$RES" | FileCheck $SRC
  if [ "$3" == "" ]; then
    rm -f $EXE $OBJ
  fi
}

if [ "$1" == "" ]; then
  for c in $ROOTDIR/lit_tests/*.{c,cc}; do
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
  for job in `jobs -p`; do
    wait $job || exit 1
  done
else
  test_file $ROOTDIR/lit_tests/$1 $CXX "DUMP"
fi
