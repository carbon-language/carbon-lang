; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "mul r0, r12, r0"  | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "mul r0, r1, r0"  | wc -l | grep 1

int %mul1(int %u) {
entry:
	%tmp = mul int %u, %u;
	ret int %tmp
}

int %mul2(int %u, int %v) {
entry:
	%tmp = mul int %u, %v;
	ret int %tmp
}
