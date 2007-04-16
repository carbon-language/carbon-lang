; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {ldr.*\\!}  | wc -l | grep 2

int *%test1(int *%X, int *%dest) {
        %Y = getelementptr int* %X, int 4
        %A = load int* %Y
        store int %A, int* %dest
        ret int* %Y
}

int %test2(int %a, int %b, int %c) {
	%tmp1 = sub int %a, %b
	%tmp2 = cast int %tmp1 to int*
	%tmp3 = load int* %tmp2
        %tmp4 = sub int %tmp1, %c
	%tmp5 = add int %tmp4, %tmp3
	ret int %tmp5
}
