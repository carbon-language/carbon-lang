
implementation

void %test1() {
	%X = malloc int*
	%Y = malloc int
	%Z = cast int* %Y to int
	%W = cast int %Z to int*
	store int* %W, int** %X
	ret void
}

void %test2(int* %P) {
	%X = malloc int*
	%Y = malloc int
	store int* %P, int** %X
	ret void
}

internal int *%test3(int* %P) {
	ret int* %P
}

void %test4() {
	%X = malloc int
	%Y = call int* %test3(int* %X)
	%ZZ = getelementptr int* null, int 17
	ret void
}
