; Check that this test makes INDVAR and related stuff dead.
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep phi | wc -l | grep 2

; XFAIL: *

declare bool %pred()

void %test1({ int, int }* %P) {
	br label %Loop
Loop:
	%INDVAR = phi int [0, %0], [%INDVAR2, %Loop]

	%gep1 = getelementptr { int, int}* %P, int %INDVAR, uint 0
	store int 0, int* %gep1

	%gep2 = getelementptr { int, int}* %P, int %INDVAR, uint 1
	store int 0, int* %gep2

	%INDVAR2 = add int %INDVAR, 1
	%cond = call bool %pred()
	br bool %cond, label %Loop, label %Out
Out:
	ret void
}

void %test2([2 x int]* %P) {
	br label %Loop
Loop:
	%INDVAR = phi int [0, %0], [%INDVAR2, %Loop]

	%gep1 = getelementptr [2 x int]* %P, int %INDVAR, uint 0
	store int 0, int* %gep1

	%gep2 = getelementptr [2 x int]* %P, int %INDVAR, uint 1
	store int 0, int* %gep2

	%INDVAR2 = add int %INDVAR, 1
	%cond = call bool %pred()
	br bool %cond, label %Loop, label %Out
Out:
	ret void
}
