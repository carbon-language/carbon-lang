; Don't reduce the byte access to P[i], at least not on targets that 
; support an efficient 'mem[r1+r2]' addressing mode.

; RUN: llvm-as < %s | opt -loop-reduce -disable-output

declare bool %pred(int)

void %test(sbyte* %PTR) {
	br label %Loop
Loop:
	%INDVAR = phi int [0, %0], [%INDVAR2, %Loop]

	%STRRED = getelementptr sbyte* %PTR, int %INDVAR
	store sbyte 0, sbyte* %STRRED

	%INDVAR2 = add int %INDVAR, 1
	%cond = call bool %pred(int %INDVAR2)  ;; cannot eliminate indvar
	br bool %cond, label %Loop, label %Out
Out:
	ret void
}
