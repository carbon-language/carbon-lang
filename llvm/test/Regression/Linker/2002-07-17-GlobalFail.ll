; RUN: llvm-as < %s > %t.bc
; RUN: echo | llvm-as > %t.tmp.bc
; RUN: llvm-link %t.tmp.bc %t.bc

%X = constant int 5
%Y = internal global [2 x int*] [ int* %X, int * %X]


