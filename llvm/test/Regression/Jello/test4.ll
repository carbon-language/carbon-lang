; test phi node
int %main() {
	br label %Test
Test:
	%X = phi int [7, %0]
	ret int %X
}
