; RUN: llvm-as < %s | opt -condprop | llvm-dis | not grep 'br label'

int %test(bool %C) {
	br bool %C, label %T1, label %F1
T1:
	br label %Cont
F1:
	br label %Cont
Cont:
	%C2 = phi bool [false, %F1], [true, %T1]
	br bool %C2, label %T2, label %F2
T2:
	call void %bar()
	ret int 17
F2:
	ret int 1
}
declare void %bar()
