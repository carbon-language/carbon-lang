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

void %testhard(int %i) {
	br label %Loop
Loop:
	%X1 = getelementptr int* %X, long 0
	%A = load int* %X1 ; Aliases X, needs to be rewritten
	%V = add int %A, 1
	%X2 = getelementptr int* %X, long 0
	store int %V, int* %X2
	br bool false, label %Loop, label %Exit

Exit:
	ret void

}
