
; RUN: as < %s > Output/%s.out1.bc
; RUN: echo "%S = type { int, int* }" | as > Output/%s.out2.bc
; RUN: link Output/%s.out[12].bc

%T = type opaque
%S = type { int, %T* }
;%X = global { int, %T* } { int 5, %T* null }
