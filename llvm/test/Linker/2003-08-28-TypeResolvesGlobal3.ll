; RUN: llvm-upgrade < %s | llvm-as > %t.out1.bc
; RUN: echo "%S = type int" | llvm-upgrade | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out2.bc %t.out1.bc

%S = type opaque

; GLobal using the resolved function prototype
global void(%S*)* %foo

void %foo(int* %V) {
  ret void
}

declare void %foo(%S*)

