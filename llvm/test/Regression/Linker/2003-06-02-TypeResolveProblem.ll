; RUN: echo "%T = type opaque" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.[12].bc

%T = type opaque
%a = constant { %T* }  { %T* null }

