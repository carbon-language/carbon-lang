// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t
// RUN: not lld -flavor gnu2 -shared %t -o %t2 2>&1 | FileCheck %s
// REQUIRES: aarch64

.hword sym+65539

// CHECK: R_AARCH64_ABS16 out of range
