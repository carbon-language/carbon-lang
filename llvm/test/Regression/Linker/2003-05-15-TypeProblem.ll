; This one fails because the LLVM runtime is allowing two null pointers of
; the same type to be created!

; RUN: echo "%S = type { %T*} %T = type opaque" | as > Output/%s.2.bc
; RUN: as < %s > Output/%s.1.bc
; RUN: link Output/%s.[12].bc

%S = type { %T* }
%T = type int

