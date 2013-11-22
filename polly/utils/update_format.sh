#!/bin/bash

CLANG_FORMAT=${CLANG_FORMAT}

if [ "${CLANG_FORMAT}x" = "x" ]; then
  CLANG_FORMAT=`which clang-format`
  if [ "${CLANG_FORMAT}x" = "x" ]; then
     echo "Error: cannot find clang-format in your path"
     exit 1
  fi
fi

for ARG in "$@"
  do
    ${CLANG_FORMAT} -i $ARG
  done
