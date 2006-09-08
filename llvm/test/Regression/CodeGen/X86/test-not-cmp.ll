; RUN: llvm-as < %s | llc -march=x86
; RUN: llvm-as < %s | llc -march=x86 | grep test

int %test(int %X, int* %y) {
	%tmp = load int* %y
	%tmp = seteq int %tmp, 0
	br bool %tmp, label %ReturnBlock, label %cond_true

cond_true:
	ret int 1

ReturnBlock:
	ret int 0
}
