// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/abs.s -o %tabs
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: not lld -flavor gnu2 %tabs %t -o %t2 2>&1 | FileCheck %s
// REQUIRES: x86

.global _start
_start:
  movl $big, %edx

#CHECK: R_X86_64_32 out of range
