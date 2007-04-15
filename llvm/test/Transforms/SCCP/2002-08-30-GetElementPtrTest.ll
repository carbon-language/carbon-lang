; RUN: llvm-upgrade < %s | llvm-as | opt -sccp | llvm-dis | not grep %X

%G = uninitialized global [40x int]

implementation

int* %test() {
	%X = getelementptr [40x int]* %G, uint 0, uint 0
	ret int* %X
}
