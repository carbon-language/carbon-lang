; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {str.*\\!}  | wc -l | grep 2

void %test1(int *%X, int *%A, int **%dest) {
        %B = load int* %A
        %Y = getelementptr int* %X, int 4
        store int %B, int* %Y
        store int* %Y, int** %dest
        ret void
}

short *%test2(short *%X, int *%A) {
        %B = load int* %A
        %Y = getelementptr short* %X, int 4
        %tmp = cast int %B to short
        store short %tmp, short* %Y
        ret short* %Y
}
