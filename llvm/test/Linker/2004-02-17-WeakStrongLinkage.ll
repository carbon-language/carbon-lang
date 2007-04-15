; RUN: llvm-upgrade < %s | llvm-as > %t.out2.bc
; RUN: echo "%me = global int* null" | llvm-upgrade | llvm-as > %t.out1.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc -o /dev/null -f

%me = weak global int * null


