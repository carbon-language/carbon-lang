; test unconditional branch
void %main() {
	br label %Test
Test:
	%X = setne int 0, 4
	br bool %X, label %Test, label %Label
Label:
	ret void
}
