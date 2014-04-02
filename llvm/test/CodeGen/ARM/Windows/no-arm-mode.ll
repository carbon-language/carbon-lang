; RUN: not llc -mtriple=armv7-windows-itanium -mcpu=cortex-a9 -o /dev/null %s 2>&1 \
; RUN:  | FileCheck %s

; CHECK: does not support ARM mode execution

