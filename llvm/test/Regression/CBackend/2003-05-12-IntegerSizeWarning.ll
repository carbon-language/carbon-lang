; Apparently this constant was unsigned in ISO C 90, but not in C 99.

int %foo() {
	ret int -2147483648
}
