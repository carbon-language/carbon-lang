// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %S/Inputs/abs.s -o %tabs
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %s -o %t
// RUN: not ld.lld %tabs %t -o /dev/null 2>&1 | FileCheck %s

.globl _start
_start:
adrp x0, big

#CHECK: R_AARCH64_ADR_PREL_PG_HI21 out of range
