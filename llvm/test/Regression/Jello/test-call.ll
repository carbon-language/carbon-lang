
declare void %exit(int)


void %FP(void(int) * %F) {
	call void %F(int 0)
	ret void
}

int %main() {
	call void %FP(void(int)* %exit)
	ret int 1
}
