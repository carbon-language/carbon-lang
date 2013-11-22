#!/bin/bash

CLANG_FORMAT=${CLANG_FORMAT}

if [ "${CLANG_FORMAT}x" = "x" ]; then
  CLANG_FORMAT=`which clang-format`
  if [ "${CLANG_FORMAT}x" = "x" ]; then
     echo "Error: cannot find clang-format in your path"
     # Disable error reporting to silence the buildbots
     exit 0
  fi
fi

OK=0

for ARG in "$@"
  do
    ${CLANG_FORMAT} $ARG | diff -u $ARG -

    if [[ $? -eq 1 ]]; then
      OK=1
    fi
  done

if [[ $OK -eq "1" ]]; then
  echo "Error: clang-format reported formatting differences"
  exit 1
else
  echo "OK: clang-format reported no formatting differences"
  exit 0
fi

