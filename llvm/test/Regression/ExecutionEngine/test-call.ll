
declare void %exit(int)

int %test(sbyte %C, short %S) {
  %X = cast short %S to int
  ret int %X
}

void %FP(void(int) * %F) {
	%X = call int %test(sbyte 123, short 1024)
	call void %F(int %X)
	ret void
}

int %main() {
	call void %FP(void(int)* %exit)
	ret int 1
}
