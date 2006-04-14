; RUN: llvm-as < %s | opt -gcse -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -gcse -instcombine | llvm-dis | not grep sub

uint %test_extractelement(<4 x uint> %V) {
        %R = extractelement <4 x uint> %V, uint 1
        %R2 = extractelement <4 x uint> %V, uint 1
	%V = sub uint %R, %R2
        ret uint %V
}

<4 x uint> %test_insertelement(<4 x uint> %V) {
        %R = insertelement <4 x uint> %V, uint 0, uint 0
        %R2 = insertelement <4 x uint> %V, uint 0, uint 0
	%x = sub <4 x uint> %R, %R2
        ret <4 x uint> %x
}

<4 x uint> %test_shufflevector(<4 x uint> %V) {
        %R = shufflevector <4 x uint> %V, <4 x uint> %V, 
                  <4 x uint> < uint 1, uint undef, uint 7, uint 2>
        %R2 = shufflevector <4 x uint> %V, <4 x uint> %V, 
                   <4 x uint> < uint 1, uint undef, uint 7, uint 2>
	%x = sub <4 x uint> %R, %R2
        ret <4 x uint> %x
}

