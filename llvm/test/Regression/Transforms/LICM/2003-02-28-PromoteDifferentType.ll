; Test that hoisting is disabled for pointers of different types...
;
; RUN: as < %s | opt -licm

void %test(int* %P) {
	br label %Loop
Loop:
	store int 5, int* %P
	%P2 = cast int* %P to sbyte*
	store sbyte 4, sbyte* %P2
	br bool true, label %Loop, label %Out
Out:
	ret void
}

