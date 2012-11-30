; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -march=ppc32 -mattr=+altivec | FileCheck %s

define <4 x i32> @test_v4i32(<4 x i32>* %X, <4 x i32>* %Y) {
	%tmp = load <4 x i32>* %X		; <<4 x i32>> [#uses=1]
	%tmp2 = load <4 x i32>* %Y		; <<4 x i32>> [#uses=1]
	%tmp3 = mul <4 x i32> %tmp, %tmp2		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp3
}
; CHECK: test_v4i32:
; CHECK: vmsumuhm
; CHECK-NOT: mullw

define <8 x i16> @test_v8i16(<8 x i16>* %X, <8 x i16>* %Y) {
	%tmp = load <8 x i16>* %X		; <<8 x i16>> [#uses=1]
	%tmp2 = load <8 x i16>* %Y		; <<8 x i16>> [#uses=1]
	%tmp3 = mul <8 x i16> %tmp, %tmp2		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %tmp3
}
; CHECK: test_v8i16:
; CHECK: vmladduhm
; CHECK-NOT: mullw

define <16 x i8> @test_v16i8(<16 x i8>* %X, <16 x i8>* %Y) {
	%tmp = load <16 x i8>* %X		; <<16 x i8>> [#uses=1]
	%tmp2 = load <16 x i8>* %Y		; <<16 x i8>> [#uses=1]
	%tmp3 = mul <16 x i8> %tmp, %tmp2		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %tmp3
}
; CHECK: test_v16i8:
; CHECK: vmuloub
; CHECK: vmuleub
; CHECK-NOT: mullw

define <4 x float> @test_float(<4 x float>* %X, <4 x float>* %Y) {
	%tmp = load <4 x float>* %X
	%tmp2 = load <4 x float>* %Y
	%tmp3 = fmul <4 x float> %tmp, %tmp2
	ret <4 x float> %tmp3
}
; Check the creation of a negative zero float vector by creating a vector of
; all bits set and shifting it 31 bits to left, resulting a an vector of 
; 4 x 0x80000000 (-0.0 as float).
; CHECK: test_float:
; CHECK: vspltisw [[ZNEG:[0-9]+]], -1
; CHECK: vslw     {{[0-9]+}}, [[ZNEG]], [[ZNEG]]
; CHECK: vmaddfp
