
declare void %exit(int)

void %main() {
	call void %exit(int 1)
	ret void
}
