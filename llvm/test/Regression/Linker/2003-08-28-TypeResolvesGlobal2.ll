; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%S = type int" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out[21].bc

%S = type opaque

void %foo(int* %V) {
  ret void
}

declare void %foo(%S*)

void %other() {
	call void %foo(%S* null)    ; Add a use of the unresolved proto
	call void %foo(int* null)   ; Add a use of the resolved function
	ret void
}
