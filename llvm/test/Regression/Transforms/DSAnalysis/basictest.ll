; very simple test
;
; RUN: opt -analyze %s -tddatastructure

implementation

int *%foo(ulong %A, double %B, long %C) {
	%X = malloc int*
	%D = cast int** %X to ulong
	%E = cast ulong %D to int*
	store int* %E, int** %X

	%F = malloc {int}
	%G = getelementptr {int}* %F, long 0, ubyte 0
	store int* %G, int** %X

	%K = malloc int **
	store int** %X, int***%K

	%H = cast long %C to int*
	ret int* null ; %H
} 

