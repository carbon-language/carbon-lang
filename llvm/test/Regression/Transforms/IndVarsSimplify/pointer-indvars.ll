; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep indvar

%G = global int* null

%Array = external global [40 x int]

void %test() {
	br label %Loop
Loop:
	%X = phi int* [getelementptr ([40 x int]* %Array, long 0, long 0), %0], [%X.next, %Loop]
	%X.next = getelementptr int* %X, long 1
	store int* %X, int** %G
	br label %Loop
}
