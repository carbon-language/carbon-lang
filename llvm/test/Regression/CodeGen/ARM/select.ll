; RUN: llvm-as < %s | llc -march=arm

int %f(int %a) {
entry:
	%tmp = seteq int %a, 4		; <bool> [#uses=1]
	%tmp1 = select bool %tmp, int 2, int 3
	ret int %tmp1
}
