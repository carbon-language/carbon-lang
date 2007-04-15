; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {ldr.*\\\[.*\],} | wc -l | grep 1

int %test(int %a, int %b, int %c) {
	%tmp1 = mul int %a, %b
	%tmp2 = cast int %tmp1 to int*
	%tmp3 = load int* %tmp2
        %tmp4 = sub int %tmp1, %c
	%tmp5 = mul int %tmp4, %tmp3
	ret int %tmp5
}
