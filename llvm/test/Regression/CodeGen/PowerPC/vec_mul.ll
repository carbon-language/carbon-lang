; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep mullw &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vmsumuhm

<4 x int> %test_v4i32(<4 x int>* %X, <4 x int>* %Y) {
        %tmp = load <4 x int>* %X
        %tmp2 = load <4 x int>* %Y
        %tmp3 = mul <4 x int> %tmp, %tmp2
        ret <4 x int> %tmp3
}

<8 x short> %test_v8i16(<8 x short>* %X, <8 x short>* %Y) {
        %tmp = load <8 x short>* %X
        %tmp2 = load <8 x short>* %Y
        %tmp3 = mul <8 x short> %tmp, %tmp2
        ret <8 x short> %tmp3
}

<16 x sbyte> %test_v16i8(<16 x sbyte>* %X, <16 x sbyte>* %Y) {
        %tmp = load <16 x sbyte>* %X
        %tmp2 = load <16 x sbyte>* %Y
        %tmp3 = mul <16 x sbyte> %tmp, %tmp2
        ret <16 x sbyte> %tmp3
}

