; RUN: opt -analyze %s -tddatastructure

void %foo(int* %X) {
	store int 4, int* %X
	ret void
}
