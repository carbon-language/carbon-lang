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

void %main() {
	br label %Test
Test:
	%X = phi int [7, %0], [%Y, %Dead]
	ret void
Dead:
	%Y = shr int 12, ubyte 4
	br label %Test
}
