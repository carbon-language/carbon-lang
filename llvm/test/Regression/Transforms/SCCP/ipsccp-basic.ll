; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | grep -v 'ret int 17' | grep -v 'ret int undef' | not grep ret

implementation

internal int %bar(int %A) {
	%X = add int 1, 2
	ret int %A
}

int %foo() {
	%X = call int %bar(int 17)
	ret int %X
}
