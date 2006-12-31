; RUN: llvm-upgrade < %s | llvm-as | opt -licm | llvm-dis

void "testfunc"(int %i.s) {

	br label %Loop

Loop:
	%j = phi uint [0, %0], [%Next, %Loop]
	%i = cast int %i.s to uint
	%i2 = mul uint %i, 17
	%Next = add uint %j, %i2
	%cond = seteq uint %Next, 0
	br bool %cond, label %Out, label %Loop

Out:
	ret void
}
