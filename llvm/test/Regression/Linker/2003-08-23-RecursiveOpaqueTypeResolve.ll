; It's a bad idea to go recursively traipsing through types without a safety 
; net.

; RUN: llvm-as < %s > Output/%s.out1.bc
; RUN: echo "%S = type { %S*, int* }" | llvm-as > Output/%s.out2.bc
; RUN: llvm-link Output/%s.out[12].bc

%S = type { %S*, opaque* }
