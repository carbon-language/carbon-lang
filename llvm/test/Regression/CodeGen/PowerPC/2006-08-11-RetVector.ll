; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vsldoi &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep vor

<4 x float> %func(<4 x float> %fp0, <4 x float> %fp1) {
        %tmp76 = shufflevector <4 x float> %fp0, <4 x float> %fp1, <4 x uint> < uint 0, uint 1, uint 2, uint 7 >                ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp76
}

