; test ret
void %main() {
BB0:
	%X = add int 1, 2
	%Y = add int %X, %X
	ret void
}
