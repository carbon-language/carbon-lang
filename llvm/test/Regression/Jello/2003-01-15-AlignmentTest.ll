
int %bar(sbyte* %X) {
	%P = alloca double   ; pointer should be 4 byte aligned!
	%R = cast double* %P to int
	%A = and int %R, 3
	ret int %A
}

int %main() {
	%SP = alloca sbyte
	%X = add uint 0, 0
	alloca sbyte, uint %X

	call int %bar(sbyte* %SP)
	ret int %0
}
