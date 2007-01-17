; RUN: llvm-upgrade < %s | llvm-as | llc -march=c



implementation

void %test() {
	%X = alloca [4xint]
	ret void
}
