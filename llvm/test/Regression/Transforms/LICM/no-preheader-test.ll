; Test that LICM works when there is not a loop-preheader
; RUN: llvm-as < %s | opt -licm | llvm-dis

void "testfunc"(int %i, bool %ifcond) {
	br bool %ifcond, label %Then, label %Else
Then:
	br label %Loop
Else:
	br label %Loop

Loop:
	%j = phi uint [0, %Then], [12, %Else], [%Next, %Loop]
	%i = cast int %i to uint
	%i2 = mul uint %i, 17
	%Next = add uint %j, %i2
	%cond = seteq uint %Next, 0
	br bool %cond, label %Out, label %Loop

Out:
	ret void
}
