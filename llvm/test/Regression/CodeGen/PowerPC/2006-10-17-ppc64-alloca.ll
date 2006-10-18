; RUN: llvm-as < %s | llc -march=ppc64

int * %foo(uint %n) {
	%A = alloca int, uint %n
	ret int* %A
}
