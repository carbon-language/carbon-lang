; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | not grep global

%G = internal global int undef

implementation

void %foo() {
	%X = load int* %G
	store int %X, int* %G
	ret void
}

int %bar() {
	%V = load int* %G
	%C = seteq int %V, 17
	br bool %C, label %T, label %F
T:
	store int 17, int* %G
	ret int %V
F:
	store int 123, int* %G
	ret int 0
}
