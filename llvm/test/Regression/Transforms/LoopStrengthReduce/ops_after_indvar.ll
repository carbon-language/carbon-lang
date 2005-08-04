; Check that this test makes INDVAR and related stuff dead, because P[indvar]
; gets reduced, making INDVAR dead.

; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | not grep INDVAR
; XFAIL: *

declare bool %pred()
declare int %getidx()

void %test([10000 x int]* %P) {
	br label %Loop
Loop:
	%INDVAR = phi int [0, %0], [%INDVAR2, %Loop]
	%idx = call int %getidx()
	%STRRED = getelementptr [10000 x int]* %P, int %INDVAR, int %idx
	store int 0, int* %STRRED

	%INDVAR2 = add int %INDVAR, 1
	%cond = call bool %pred()
	br bool %cond, label %Loop, label %Out
Out:
	ret void
}
