; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep phi | wc -l | grep 1

; This testcase should have ONE stride 18 indvar, the other use should have a
; loop invariant value (B) added to it inside of the loop, instead of having
; a whole indvar based on B for it.

; XFAIL: *

declare bool %cond(uint)

void %test(uint %B) {
	br label %Loop
Loop:
	%IV = phi uint [0, %0], [%IVn, %Loop]

	%C = mul uint %IV, 18
	%D = mul uint %IV, 18
	%E = add uint %D, %B

	%cnd = call bool %cond(uint %E)
	call bool %cond(uint %C)
	%IVn = add uint %IV, 1
	br bool %cnd, label %Loop, label %Out
Out:
	ret void
}
