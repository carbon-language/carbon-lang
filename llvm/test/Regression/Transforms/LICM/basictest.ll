; RUN: llvm-as < %s | opt -licm | llvm-dis

void "testfunc"(int %i) {

	br label %Loop

Loop:
	%j = phi uint [0, %0], [%Next, %Loop]
	%i = cast int %i to uint
	%i2 = mul uint %i, 17
	%Next = add uint %j, %i2
	%cond = seteq uint %Next, 0
	br bool %cond, label %Out, label %Loop

Out:
	ret void
}
