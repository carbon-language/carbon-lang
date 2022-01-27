# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "SECTIONS { patatino = 0x1234; }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-nm %t1 | FileCheck %s
# CHECK: 0000000000001234 A patatino

.global _start
_start:
  call patatino
