; RUN: llvm-as < %s | opt -mem2reg | llvm-dis | not grep phi

implementation

int %testfunc(bool %C, int %i, sbyte %j) {
	%I = alloca int
	br bool %C, label %T, label %Cont
T:
	store int %i, int* %I
	br label %Cont
Cont:
	%Y = load int* %I  ;; %Y = phi %i, undef -> %Y = %i
	ret int %Y
}
