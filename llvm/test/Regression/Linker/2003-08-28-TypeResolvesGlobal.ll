; RUN: llvm-as < %s > Output/%s.out1.bc
; RUN: echo "%S = type int" | llvm-as > Output/%s.out2.bc
; RUN: llvm-link Output/%s.out[21].bc

%S = type opaque

void %foo(int* %V) {
  ret void
}

declare void %foo(%S*)

