; RUN: as < %s | opt -sccp | dis | not grep '%X'

%G = uninitialized global [40x int]

implementation

int* %test() {
	%X = getelementptr [40x int]* %G, uint 0, uint 0
	ret int* %X
}
