; This one fails because the LLVM runtime is allowing two null pointers of
; the same type to be created!

; RUN: echo {%S = type \{ %T*\} %T = type opaque} | llvm-as > %t.2.bc
; RUN: llvm-upgrade < %s | llvm-as > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc

%S = type { %T* }
%T = type int

