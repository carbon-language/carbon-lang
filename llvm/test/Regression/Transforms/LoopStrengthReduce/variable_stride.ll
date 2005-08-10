; Check that variable strides are reduced to adds instead of multiplies.
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | not grep 'mul'

declare bool %pred(int)

void %test([10000 x int]* %P, int %STRIDE) {
	br label %Loop
Loop:
	%INDVAR = phi int [0, %0], [%INDVAR2, %Loop]
	%Idx = mul int %INDVAR, %STRIDE

	%cond = call bool %pred(int %Idx)
	%INDVAR2 = add int %INDVAR, 1
	br bool %cond, label %Loop, label %Out
Out:
	ret void
}
