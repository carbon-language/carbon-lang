// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %tout
// RUN: llvm-objdump -d %tout | FileCheck %s
// REQUIRES: x86

.global _start
_start:
  movl $1, sym1(%rip)

.weak sym1

// CHECK: movl    $1, -69642(%rip)
