; Check that this test makes INDVAR and related stuff dead.
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | not grep INDVAR

declare bool %pred()

void %test(int* %P) {
	br label %Loop
Loop:
	%INDVAR = phi int [0, %0], [%INDVAR2, %Loop]

	%STRRED = getelementptr int* %P, int %INDVAR
	store int 0, int* %STRRED

	%INDVAR2 = add int %INDVAR, 1
	%cond = call bool %pred()
	br bool %cond, label %Loop, label %Out
Out:
	ret void
}
