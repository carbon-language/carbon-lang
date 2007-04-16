; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep cmp  | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep test | wc -l | grep 1

int %f1(int %X, int* %y) {
	%tmp = load int* %y
	%tmp = seteq int %tmp, 0
	br bool %tmp, label %ReturnBlock, label %cond_true

cond_true:
	ret int 1

ReturnBlock:
	ret int 0
}

int %f2(int %X, int* %y) {
	%tmp = load int* %y
        %tmp1 = shl int %tmp, ubyte 3
	%tmp1 = seteq int %tmp1, 0
	br bool %tmp1, label %ReturnBlock, label %cond_true

cond_true:
	ret int 1

ReturnBlock:
	ret int 0
}
