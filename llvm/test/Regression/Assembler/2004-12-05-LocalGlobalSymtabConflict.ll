; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep '%G = alloca int'

; In this testcase, %bar stores to the global G.  Make sure that inlining does
; not cause it to store to the G in main instead.

%G = global int 7

int %main() {
	%G = alloca int
	store int 0, int* %G
	call void %bar()
	%RV = load int* %G
	ret int %RV
}

internal void %bar() {
	store int 123, int* %G
	ret void
}

