

int %main() {
	br label %Loop
Loop:
	%I = phi int [0, %0], [%i2, %Loop]
	%i2 = add int %I, 1
	%C = seteq int %i2, 10
	br bool %C, label %Out, label %Loop
Out:
	ret int 0
}
