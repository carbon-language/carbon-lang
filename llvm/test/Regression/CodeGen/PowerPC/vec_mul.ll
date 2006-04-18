; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep mullw
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vmsumuhm

<4 x int> %test(<4 x int>* %X, <4 x int>* %Y) {
        %tmp = load <4 x int>* %X
        %tmp2 = load <4 x int>* %Y
        %tmp3 = mul <4 x int> %tmp, %tmp2
        ret <4 x int> %tmp3
}

