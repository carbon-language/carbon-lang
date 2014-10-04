; RUN: llc < %s -march=x86 -mcpu=core2 -mattr=+ssse3 | FileCheck %s
; RUN: llc < %s -march=x86 -mcpu=yonah | FileCheck --check-prefix=CHECK-YONAH %s

define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test1:
; CHECK:       # BB#0:
; CHECK-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,2,3,0]
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test1:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,2,3,0]
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> < i32 1, i32 2, i32 3, i32 0 >
	ret <4 x i32> %C
}

define <4 x i32> @test2(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test2:
; CHECK:       # BB#0:
; CHECK-NEXT:    palignr {{.*#+}} xmm1 = xmm0[4,5,6,7,8,9,10,11,12,13,14,15],xmm1[0,1,2,3]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test2:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,0],xmm0[3,0]
; CHECK-YONAH-NEXT:    shufps {{.*#+}} xmm0 = xmm0[1,2],xmm1[2,0]
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> < i32 1, i32 2, i32 3, i32 4 >
	ret <4 x i32> %C
}

define <4 x i32> @test3(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test3:
; CHECK:       # BB#0:
; CHECK-NEXT:    palignr {{.*#+}} xmm1 = xmm0[4,5,6,7,8,9,10,11,12,13,14,15],xmm1[0,1,2,3]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test3:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    shufps {{.*#+}} xmm0 = xmm0[1,2],xmm1[2,0]
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> < i32 1, i32 2, i32 undef, i32 4 >
	ret <4 x i32> %C
}

define <4 x i32> @test4(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: test4:
; CHECK:       # BB#0:
; CHECK-NEXT:    palignr {{.*#+}} xmm0 = xmm1[8,9,10,11,12,13,14,15],xmm0[0,1,2,3,4,5,6,7]
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test4:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1],xmm0[0]
; CHECK-YONAH-NEXT:    movapd %xmm1, %xmm0
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> < i32 6, i32 7, i32 undef, i32 1 >
	ret <4 x i32> %C
}

define <4 x float> @test5(<4 x float> %A, <4 x float> %B) nounwind {
; CHECK-LABEL: test5:
; CHECK:       # BB#0:
; CHECK-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1],xmm0[0]
; CHECK-NEXT:    movapd %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test5:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1],xmm0[0]
; CHECK-YONAH-NEXT:    movapd %xmm1, %xmm0
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 6, i32 7, i32 undef, i32 1 >
	ret <4 x float> %C
}

define <8 x i16> @test6(<8 x i16> %A, <8 x i16> %B) nounwind {
; CHECK-LABEL: test6:
; CHECK:       # BB#0:
; CHECK-NEXT:    palignr {{.*#+}} xmm1 = xmm0[6,7,8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test6:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[2,3,0,1]
; CHECK-YONAH-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm1 = xmm1[0,2,2,3,4,5,6,7]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,7,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm1 = xmm1[3,0,1,2,4,5,6,7]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,6,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,1,2,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[3,0,2,1,4,5,6,7]
; CHECK-YONAH-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 3, i32 4, i32 undef, i32 6, i32 7, i32 8, i32 9, i32 10 >
	ret <8 x i16> %C
}

define <8 x i16> @test7(<8 x i16> %A, <8 x i16> %B) nounwind {
; CHECK-LABEL: test7:
; CHECK:       # BB#0:
; CHECK-NEXT:    palignr {{.*#+}} xmm1 = xmm0[10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7,8,9]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test7:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; CHECK-YONAH-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,2,2,1,4,5,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[3,1,2,0]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,7,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,1,2,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm1 = xmm1[1,2,3,0,4,5,6,7]
; CHECK-YONAH-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 undef, i32 6, i32 undef, i32 8, i32 9, i32 10, i32 11, i32 12 >
	ret <8 x i16> %C
}

define <16 x i8> @test8(<16 x i8> %A, <16 x i8> %B) nounwind {
; CHECK-LABEL: test8:
; CHECK:       # BB#0:
; CHECK-NEXT:    palignr {{.*#+}} xmm1 = xmm0[5,6,7,8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test8:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    pxor %xmm3, %xmm3
; CHECK-YONAH-NEXT:    movdqa %xmm0, %xmm2
; CHECK-YONAH-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[2,3,2,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm2 = xmm2[1,2,3,3,4,5,6,7]
; CHECK-YONAH-NEXT:    punpckhbw {{.*#+}} xmm0 = xmm0[8],xmm3[8],xmm0[9],xmm3[9],xmm0[10],xmm3[10],xmm0[11],xmm3[11],xmm0[12],xmm3[12],xmm0[13],xmm3[13],xmm0[14],xmm3[14],xmm0[15],xmm3[15]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[3,1,2,0]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm4 = xmm4[0,1,2,3,4,7,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm4 = xmm4[2,1,2,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm4 = xmm4[1,2,3,0,4,5,6,7]
; CHECK-YONAH-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm4[0]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[1,2,3,3,4,5,6,7]
; CHECK-YONAH-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[3,1,2,0]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,7,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,3,2,1]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm1 = xmm1[0,1,2,2,4,5,6,7]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,5,6,7,4]
; CHECK-YONAH-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,2,2,3,4,5,6,7]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,7,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; CHECK-YONAH-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-YONAH-NEXT:    packuswb %xmm0, %xmm2
; CHECK-YONAH-NEXT:    movdqa %xmm2, %xmm0
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <16 x i8> %A, <16 x i8> %B, <16 x i32> < i32 5, i32 6, i32 7, i32 undef, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20 >
	ret <16 x i8> %C
}

; Check that we don't do unary (circular on single operand) palignr incorrectly.
; (It is possible, but before this testcase was committed, it was being done
; incorrectly.  In particular, one of the operands of the palignr node
; was an UNDEF.)
define <8 x i16> @test9(<8 x i16> %A, <8 x i16> %B) nounwind {
; CHECK-LABEL: test9:
; CHECK:       # BB#0:
; CHECK-NEXT:    palignr {{.*#+}} xmm1 = xmm1[2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; CHECK-YONAH-LABEL: test9:
; CHECK-YONAH:       # BB#0:
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,2,1,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,3,2,3,4,5,6,7]
; CHECK-YONAH-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,1,0,3]
; CHECK-YONAH-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,0,1,2,4,5,6,7]
; CHECK-YONAH-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,5,6,7,4]
; CHECK-YONAH-NEXT:    retl
  %C = shufflevector <8 x i16> %B, <8 x i16> %A, <8 x i32> < i32 undef, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0 >
	ret <8 x i16> %C
}

