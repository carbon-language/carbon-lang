; test phi node
void %main() {
	br label %Test
Test:
	%X = phi int [7, %0], [%Y, %Dead]
	ret void
Dead:
	%Y = shr int 12, ubyte 4
	br label %Test
}
