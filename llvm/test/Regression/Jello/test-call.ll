
declare void %foo()

void %test1() {
	call void %foo()
	ret void
}
