
declare void %exit(int)

int %main() {
	call void %exit(int 0)
	ret int 1
}
