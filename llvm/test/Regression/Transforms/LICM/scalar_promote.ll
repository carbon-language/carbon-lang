; RUN: as < %s | opt  -licm -stats 2>&1 | grep "memory locations promoted to register"

%X = global int 7

void %testfunc(int %i) {
	br label %Loop

Loop:
	%j = phi uint [0, %0], [%Next, %Loop]

	%x = load int* %X  ; Should promote this to a register inside of loop!
	%x2 = add int %x, 1
	store int %x2, int* %X

	%Next = add uint %j, 1
	%cond = seteq uint %Next, 0
	br bool %cond, label %Out, label %Loop

Out:
	ret void
}
