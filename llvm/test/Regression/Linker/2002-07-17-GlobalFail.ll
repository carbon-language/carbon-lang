; RUN: as < %s > Output/%s.bc
; RUN: echo | as > Output/%s.tmp.bc
; RUN: link Output/%s.tmp.bc Output/%s.bc

%X = constant int 5
%Y = internal global [2 x int*] [ int* %X, int * %X]


