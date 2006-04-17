; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep vperm

<4 x float> %test_uu72(<4 x float> *%P1, <4 x float> *%P2) {
	%V1 = load <4 x float> *%P1
	%V2 = load <4 x float> *%P2
	; vmrglw + vsldoi
	%V3 = shufflevector <4 x float> %V1, <4 x float> %V2,
	                    <4 x uint> <uint undef, uint undef, uint 7, uint 2>
	ret <4 x float> %V3
}

<4 x float> %test_30u5(<4 x float> *%P1, <4 x float> *%P2) {
	%V1 = load <4 x float> *%P1
	%V2 = load <4 x float> *%P2
	%V3 = shufflevector <4 x float> %V1, <4 x float> %V2,
	          <4 x uint> <uint 3, uint 0, uint undef, uint 5>
	ret <4 x float> %V3
}

<4 x float> %test_3u73(<4 x float> *%P1, <4 x float> *%P2) {
	%V1 = load <4 x float> *%P1
	%V2 = load <4 x float> *%P2
	%V3 = shufflevector <4 x float> %V1, <4 x float> %V2,
	          <4 x uint> <uint 3, uint undef, uint 7, uint 3>
	ret <4 x float> %V3
}

<4 x float> %test_3774(<4 x float> *%P1, <4 x float> *%P2) {
	%V1 = load <4 x float> *%P1
	%V2 = load <4 x float> *%P2
	%V3 = shufflevector <4 x float> %V1, <4 x float> %V2,
	          <4 x uint> <uint 3, uint 7, uint 7, uint 4>
	ret <4 x float> %V3
}

<4 x float> %test_4450(<4 x float> *%P1, <4 x float> *%P2) {
	%V1 = load <4 x float> *%P1
	%V2 = load <4 x float> *%P2
	%V3 = shufflevector <4 x float> %V1, <4 x float> %V2,
	          <4 x uint> <uint 4, uint 4, uint 5, uint 0>
	ret <4 x float> %V3
}
