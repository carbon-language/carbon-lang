; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo { %M = type \[8 x i32\] external global %M } | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc -S | grep %M | grep \\{
%M = type { i32 }


