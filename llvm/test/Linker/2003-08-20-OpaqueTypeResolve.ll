; RUN: llvm-upgrade < %s | llvm-as > %t.out1.bc
; RUN: echo {%S = type \{ int, int* \} } | llvm-upgrade | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc

%T = type opaque
%S = type { int, %T* }
;%X = global { int, %T* } { int 5, %T* null }
