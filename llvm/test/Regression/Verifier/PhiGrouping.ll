

int "test"(int %i, int %j, bool %c) {
	br bool %c, label %A, label %B
A:
	br label %C
B:
	br label %C

C:
	%a = phi int [%i, %A], [%j, %B]
	%x = add int %a, 0                 ; Error, PHI's should be grouped!
	%b = phi int [%i, %A], [%j, %B]
	ret int %x
}
