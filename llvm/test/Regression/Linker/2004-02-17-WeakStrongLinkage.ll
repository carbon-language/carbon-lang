; RUN: llvm-as < %s > %t.out2.bc
; RUN: echo "%me = global int* null" | llvm-as > %t.out1.bc
; RUN: llvm-link %t.out[12].bc -o /dev/null -f

%me = weak global int * null


