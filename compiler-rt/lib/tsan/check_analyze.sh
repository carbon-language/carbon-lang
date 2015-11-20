#!/bin/bash
set -u

RES=$(./analyze_libtsan.sh)
PrintRes() {
  printf "%s\n" "$RES"
}

PrintRes

check() {
  res=$(PrintRes | egrep "$1 .* $2 $3; ")
  if [ "$res" == "" ]; then
    echo FAILED $1 must contain $2 $3
    exit 1
  fi
}

for f in write1; do
  check $f rsp 1
  check $f push 2
  check $f pop 2
done

for f in write2 write4; do
  check $f rsp 1
  check $f push 4
  check $f pop 4
done

for f in write8; do
  check $f rsp 1
  check $f push 3
  check $f pop 3
done

for f in read1 read2 read4 read8; do
  check $f rsp 1
  check $f push 5
  check $f pop 5
done

for f in func_entry func_exit; do
  check $f rsp 0
  check $f push 0
  check $f pop 0
  check $f call 1  # TraceSwitch()
done

echo LGTM
