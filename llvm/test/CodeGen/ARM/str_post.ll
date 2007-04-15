; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {strh .*\\\[.*\], #-4}  | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {str .*\\\[.*\],}  | wc -l | grep 1

short %test1(int *%X, short *%A) {
        %Y = load int* %X
        %tmp1 = cast int %Y to short
        store short %tmp1, short* %A
        %tmp2 = cast short* %A to short
        %tmp3 = sub short %tmp2, 4
        ret short %tmp3
}

int %test2(int *%X, int *%A) {
        %Y = load int* %X
        store int %Y, int* %A
        %tmp1 = cast int* %A to int
        %tmp2 = sub int %tmp1, 4
        ret int %tmp2
}
