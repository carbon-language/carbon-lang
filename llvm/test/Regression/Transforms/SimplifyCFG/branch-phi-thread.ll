; RUN: llvm-as < %s | opt -simplifycfg -adce | llvm-dis | not grep 'call void %f1' &&
; RUN: llvm-as < %s | opt -simplifycfg -adce -disable-output
declare void %f1()
declare void %f2()
declare void %f3()
declare void %f4()

implementation

int %test1(int %X, bool %D) {
E:
	%C = seteq int %X, 0
	br bool %C, label %T, label %F
T:
	br bool %C, label %B, label %A
A:
	call void %f1()
	br bool %D, label %T, label %F
B:
	call void %f2()
	ret int 345
F:
	call void %f3()
	ret int 123
}

int %test2(int %X, bool %D) {
E:
	%C = seteq int %X, 0
	br bool %C, label %T, label %F
T:
	%P = phi bool [true, %E], [%C, %A]
	br bool %P, label %B, label %A
A:
	call void %f1()
	br bool %D, label %T, label %F
B:
	call void %f2()
	ret int 345
F:
	call void %f3()
	ret int 123
}

int %test3(int %X, bool %D, int* %AP, int* %BP) {
E:
	%C = seteq int %X, 0
	br bool %C, label %T, label %F
T:
	call void %f3()  ;; Inst in block.
	%XX = load int* %AP
	store int %XX, int* %BP
	br bool %C, label %B, label %A
A:
	call void %f1()
	br bool %D, label %T, label %F
B:
	call void %f2()
	ret int 345
F:
	call void %f3()
	ret int 123
}
