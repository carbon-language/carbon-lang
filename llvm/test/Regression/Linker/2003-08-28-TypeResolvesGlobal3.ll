; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%S = type int" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out[21].bc

%S = type opaque

; GLobal using the resolved function prototype
global void(%S*)* %foo

void %foo(int* %V) {
  ret void
}

declare void %foo(%S*)

