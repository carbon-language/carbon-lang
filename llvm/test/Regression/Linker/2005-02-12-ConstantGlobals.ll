; Test that a prototype can be marked const, and the definition is allowed
; to be nonconst.

; RUN: echo "%X = global int 7" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep constant

%X = external constant int
