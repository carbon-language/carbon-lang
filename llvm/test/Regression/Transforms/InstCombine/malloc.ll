; test that malloc's with a constant argument are promoted to array allocations
; RUN: as < %s | opt -instcombine -die | dis | grep getelementptr

int* %test() {
	%X = malloc int, uint 4
	ret int* %X
}
