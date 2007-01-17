; Test that a prototype can be marked const, and the definition is allowed
; to be nonconst.

; RUN: echo "%X = global int 7" | llvm-upgrade | llvm-as > %t.2.bc
; RUN: llvm-upgrade < %s | llvm-as > %t.1.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep 'global i32 7'

%X = external constant int
