; It's a bad idea to go recursively trapesing through types without a safety 
; net.

; RUN: as < %s > Output/%s.out1.bc
; RUN: echo "%S = type { %S*, int* }" | as > Output/%s.out2.bc
; RUN: link Output/%s.out[12].bc

%S = type { %S*, opaque* }
