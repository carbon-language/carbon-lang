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

int %main() {
	br label %Test
Test:
	%X = phi int [7, %0], [%Y, %Dead]
	ret int 0
Dead:
	%Y = shr int 12, ubyte 4
	br label %Test
}
