; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep mul | wc -l | grep 2

int %f1(int %u) {
entry:
	%tmp = mul int %u, %u;
	ret int %tmp
}

int %f2(int %u, int %v) {
entry:
	%tmp = mul int %u, %v;
	ret int %tmp
}
