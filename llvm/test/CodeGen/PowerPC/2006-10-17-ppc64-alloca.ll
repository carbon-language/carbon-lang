; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64

int * %foo(uint %n) {
	%A = alloca int, uint %n
	ret int* %A
}
