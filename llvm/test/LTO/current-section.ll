; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1

module asm ".align 4"
