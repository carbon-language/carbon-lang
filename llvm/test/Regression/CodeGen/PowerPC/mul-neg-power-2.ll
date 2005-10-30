; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep mul

int %test1(int %a) {
        %tmp.1 = mul int %a, -2         ; <int> [#uses=1]
        %tmp.2 = add int %tmp.1, 63             ; <int> [#uses=1]
        ret int %tmp.2
}

