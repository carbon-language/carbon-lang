; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | grep -v 'ret int 0' | grep -v 'ret int undef' | not grep ret

implementation

internal int %bar(int %A) {
	%C = seteq int %A, 0
	br bool %C, label %T, label %F
T:
	%B = call int %bar(int 0)
	ret int 0
F:      ; unreachable
	%C = call int %bar(int 1)
	ret int %C
}

int %foo() {
	%X = call int %bar(int 0)
	ret int %X
}
