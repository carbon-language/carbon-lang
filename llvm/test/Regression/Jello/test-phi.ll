; test phi node

%Y = global int 6

void %blah(int *%X) {
	br label %T
T:
	phi int* [%X, %0], [%Y, %Dead]
	ret void
Dead:
	br label %T
}

int %test(bool %C) {
	br bool %C, label %T, label %T
T:
	%X = phi int [123, %0], [123, %0]
	ret int %X
}

int %main() {
	br label %Test
Test:
	%X = phi int [0, %0], [%Y, %Dead]
	ret int %X
Dead:
	%Y = shr int 12, ubyte 4
	br label %Test
}
