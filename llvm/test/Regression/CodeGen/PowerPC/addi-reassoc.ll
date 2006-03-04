; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep addi

        %struct.X = type { [5 x sbyte] }
implementation   ; Functions:

int %test1([4 x int]* %P, int %i) {
        %tmp.2 = add int %i, 2          ; <int> [#uses=1]
        %tmp.4 = getelementptr [4 x int]* %P, int %tmp.2, int 1
        %tmp.5 = load int* %tmp.4
        ret int %tmp.5
}

int %test2(%struct.X* %P, int %i) {
        %tmp.2 = add int %i, 2
        %tmp.5 = getelementptr %struct.X* %P, int %tmp.2, uint 0, int 1
        %tmp.6 = load sbyte* %tmp.5
        %tmp.7 = cast sbyte %tmp.6 to int
        ret int %tmp.7
}

