; This is the same testcase as 2003-06-29-NodeCollapsing2.ll, but it uses the 
; graph checker.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -datastructure-gc -dsgc-abort-if-any-collapsed
;
%T = type { int}

int %main() {
	%A = alloca %T
	%B = alloca { %T }
	%C = alloca %T*
	%Bp = getelementptr { %T }* %B, long 0, uint 0
	%Ap = getelementptr %T* %A, long 0, uint 0

	store %T* %A, %T** %C
	store %T* %Bp, %T** %C    ; This store was causing merging to happen!
	ret int 0
}
