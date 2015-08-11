// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/resolution.s -o %t2
// RUN: lld -flavor gnu2 %t %t2 -o %t3
// REQUIRES: x86

.globl _start
_start:
        nop

local:

.weak foo
foo:

.long bar
