
declare int %callee(int, int)


int %test(int %X) {
	%A = invoke int %callee(int %X, int 5) to label %Ok except label %Threw
Ok:
	%B = phi int [%A, %0], [-1, %Threw]
	ret int %A
Threw:
	br label %Ok
}
