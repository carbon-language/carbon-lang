; RUN: as < %s > Output/%s.out1.bc
; RUN: echo "%S = type int" | as > Output/%s.out2.bc
; RUN: link Output/%s.out[21].bc

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
