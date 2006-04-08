; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

uint %test_extractelement(<4 x uint> %V) {
        %R = extractelement <4 x uint> %V, uint 1
        ret uint %R
}

<4 x uint> %test_insertelement(<4 x uint> %V) {
        %R = insertelement <4 x uint> %V, uint 0, uint 0
        ret <4 x uint> %R
}

<4 x uint> %test_shufflevector(<4 x uint> %V) {
        %R = shufflevector <4 x uint> %V, <4 x uint> %V, 
                  <4 x uint> < uint 1, uint undef, uint 7, uint 2>
        ret <4 x uint> %R
}

<4 x float> %test_shufflevector(<4 x float> %V) {
        %R = shufflevector <4 x float> %V, <4 x float> undef, 
                  <4 x uint> < uint 1, uint undef, uint 7, uint 2>
        ret <4 x float> %R
}
