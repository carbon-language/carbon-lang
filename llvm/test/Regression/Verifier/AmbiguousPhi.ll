; RUN: not llvm-as -f %s -o /dev/null



int "test"(int %i, int %j, bool %c) {
	br bool %c, label %A, label %A
A:
	%a = phi int [%i, %0], [%j, %0]  ; Error, different values from same block!
	ret int %a
}
