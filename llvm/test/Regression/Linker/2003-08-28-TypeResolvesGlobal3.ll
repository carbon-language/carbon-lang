; RUN: as < %s > Output/%s.out1.bc
; RUN: echo "%S = type int" | as > Output/%s.out2.bc
; RUN: link Output/%s.out[21].bc

%S = type opaque

; GLobal using the resolved function prototype
global void(%S*)* %foo

void %foo(int* %V) {
  ret void
}

declare void %foo(%S*)

