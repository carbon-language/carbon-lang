; This testcases makes sure that mem2reg can handle unreachable blocks.
; RUN: as < %s | opt -mem2reg

int %test() {
	%X = alloca int

	store int 6, int* %X
	br label %Loop
Loop:
	store int 5, int* %X
	br label %EndOfLoop
Unreachable:
	br label %EndOfLoop

EndOfLoop:
	br label %Loop
}
