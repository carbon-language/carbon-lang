; This testcase is invalid because we are indexing into a pointer that is 
; contained WITHIN a structure.

void %test({int, int*} * %X) {
	getelementptr {int, int*} * %X, long 0, ubyte 1, long 0
	ret void
}
