; RUN: llvm-as < %s | llc -march=c



implementation

void %test() {
	%X = alloca [4xint]
	ret void
}
