
; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%S = type { int, int* }" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out[12].bc

%T = type opaque
%S = type { int, %T* }
;%X = global { int, %T* } { int 5, %T* null }
