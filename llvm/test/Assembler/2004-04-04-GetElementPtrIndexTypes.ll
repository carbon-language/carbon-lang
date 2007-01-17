; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

int *%t1({ float, int }* %X) {
	%W = getelementptr { float, int }* %X, int 20, uint 1
	%X = getelementptr { float, int }* %X, uint 20, uint 1
	%Y = getelementptr { float, int }* %X, long 20, uint 1
	%Z = getelementptr { float, int }* %X, ulong 20, uint 1
	ret int* %Y
}
