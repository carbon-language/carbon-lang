; Mem2reg used to only add one incoming value to a PHI node, even if it had
; multiple incoming edges from a block.
;
; RUN: llvm-as < %s | opt -mem2reg -disable-output

int %test(bool %c1, bool %c2) {
	%X = alloca int
	br bool %c1, label %Exit, label %B2
B2:
	store int 2, int* %X
	br bool %c2, label %Exit, label %Exit
Exit:
	%Y = load int *%X
	ret int %Y
}
