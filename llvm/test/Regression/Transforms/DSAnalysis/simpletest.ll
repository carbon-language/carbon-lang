; RUN: analyze %s -tddatastructure

implementation

int *%foo(int *%A, int **%B, int *%C, int **%D, int* %E) {
	%a = load int* %A
	%b = load int** %B

	store int* %C, int** %D

	ret int* %E
}
