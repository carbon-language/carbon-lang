; RUN: echo "%T = type int" | llvm-as > %t.1.bc
; RUN: llvm-as < %s > %t.2.bc
; RUN: llvm-link %t.[12].bc

%T = type opaque

%X = constant {%T*} {%T* null }

