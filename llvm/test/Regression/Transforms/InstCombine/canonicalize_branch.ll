; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 'setne\|setle\|setge'

int %test1(uint %X, uint %Y) {
	%C = setne uint %X, %Y
	br bool %C, label %T, label %F
T:
	ret int 12
F:
	ret int 123
}

int %test2(uint %X, uint %Y) {
	%C = setle uint %X, %Y
	br bool %C, label %T, label %F
T:
	ret int 12
F:
	ret int 123
}
int %test3(uint %X, uint %Y) {
	%C = setge uint %X, %Y
	br bool %C, label %T, label %F
T:
	ret int 12
F:
	ret int 123
}
