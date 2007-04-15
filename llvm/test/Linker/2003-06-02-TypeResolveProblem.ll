; RUN: echo {%T = type opaque} | llvm-upgrade | llvm-as > %t.2.bc
; RUN: llvm-upgrade < %s | llvm-as > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc

%T = type opaque
%a = constant { %T* }  { %T* null }

