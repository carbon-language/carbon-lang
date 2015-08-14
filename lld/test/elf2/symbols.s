// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-readobj -symbols %t2 | FileCheck %s
// REQUIRES: x86

.globl _start;
_start:

// CHECK: Symbols [
// CHECK-NEXT: ]
