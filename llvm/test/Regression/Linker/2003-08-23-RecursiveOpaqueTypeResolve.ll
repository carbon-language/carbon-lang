; It's a bad idea to go recursively traipsing through types without a safety 
; net.

; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%S = type { %S*, int* }" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out[12].bc

%S = type { %S*, opaque* }
