#!/bin/bash
set -u

RES=$(./analyze_libtsan.sh)
PrintRes() {
  printf "%s\n" "$RES"
}

PrintRes

mops="write1 \
      write2 \
      write4 \
      write8 \
      read1 \
      read2 \
      read4 \
      read8"
func="func_entry \
      func_exit"

check() {
  res=$(PrintRes | egrep "$1 .* $2 $3; ")
  if [ "$res" == "" ]; then
    echo FAILED $1 must contain $2 $3
    exit 1
  fi
}

for f in $mops; do
  check $f rsp 1   # To read caller pc.
  check $f push 0
  check $f pop 0
done

for f in $func; do
  check $f rsp 0
  check $f push 0
  check $f pop 0
  check $f call 1  # TraceSwitch()
done

echo LGTM
