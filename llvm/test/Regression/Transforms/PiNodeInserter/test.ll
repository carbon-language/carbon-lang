
int "test"(int %i) {
	%c = seteq int %i, 0
	br bool %c, label %iIsZero, label %iIsNotZero

iIsZero:
	ret int %i

iIsZero2:
	ret int 0
iIsNotZero:
	%d = setne int %i, 0
	br bool %d, label %Quit, label %iIsZero2

Quit:
	ret int 1
}
