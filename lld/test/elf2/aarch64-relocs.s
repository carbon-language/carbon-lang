# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t
# RUN: lld -flavor gnu2 %t -o %t2
# RUN: llvm-objdump -d %t2 | FileCheck %s
# REQUIRES: aarch64

.globl _start
_start:
  adr x1,msg
msg:  .asciz  "Hello, world\n"
msgend:

// R_AARCH64_ADR_PREL_LO21
# CHECK: _start:
# CHECK:   11000:       21 00 00 10     adr     x1, #4
