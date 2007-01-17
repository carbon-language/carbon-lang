; RUN: llvm-upgrade < %s | llvm-as > %t.out1.bc
; RUN: echo "%S = type [8 x int] external global %S " | llvm-upgrade | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out[12].bc | llvm-dis | grep %S | grep '{'

%S = type { int }


