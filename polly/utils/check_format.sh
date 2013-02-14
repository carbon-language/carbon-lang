#!/bin/bash

OK=0

for ARG in "$@"
  do
    clang-format $ARG | diff -u $ARG -

    if [[ $? -eq 1 ]]; then
      OK=1
    fi
  done

if [[ $OK -eq "1" ]]; then
  echo "Error: clang-format reported formatting differences"
else
  echo "OK: clang-format reported no formatting differences"
fi

