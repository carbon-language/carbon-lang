; RUN: llvm-upgrade < %s | llvm-as > %t.out1.bc
; RUN: echo "%S = type int" | llvm-upgrade | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out[21].bc

%S = type opaque

void %foo(int* %V) {
  ret void
}

declare void %foo(%S*)

