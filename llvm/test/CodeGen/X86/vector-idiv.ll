; RUN: llc -march=x86-64 -mcpu=core2 -mattr=+sse4.1 < %s | FileCheck %s -check-prefix=SSE41
; RUN: llc -march=x86-64 -mcpu=core2 < %s | FileCheck %s -check-prefix=SSE
; RUN: llc -march=x86-64 -mcpu=core-avx2 < %s | FileCheck %s -check-prefix=AVX

define <4 x i32> @test1(<4 x i32> %a) {
  %div = udiv <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %div

; SSE41-LABEL: test1:
; SSE41: pmuludq
; SSE41: pshufd	$49
; SSE41: pmuludq
; SSE41: shufps	$-35
; SSE41: psubd
; SSE41: psrld $1
; SSE41: padd
; SSE41: psrld $2

; AVX-LABEL: test1:
; AVX: vpmuludq
; AVX: vpshufd	$49
; AVX: vpmuludq
; AVX: vshufps	$-35
; AVX: vpsubd
; AVX: vpsrld $1
; AVX: vpadd
; AVX: vpsrld $2
}

define <8 x i32> @test2(<8 x i32> %a) {
  %div = udiv <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div

; AVX-LABEL: test2:
; AVX: vpbroadcastd
; AVX: vpalignr $4
; AVX: vpmuludq
; AVX: vpmuludq
; AVX: vpblendd $170
; AVX: vpsubd
; AVX: vpsrld $1
; AVX: vpadd
; AVX: vpsrld $2
}

define <8 x i16> @test3(<8 x i16> %a) {
  %div = udiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div

; SSE41-LABEL: test3:
; SSE41: pmulhuw
; SSE41: psubw
; SSE41: psrlw $1
; SSE41: paddw
; SSE41: psrlw $2

; AVX-LABEL: test3:
; AVX: vpmulhuw
; AVX: vpsubw
; AVX: vpsrlw $1
; AVX: vpaddw
; AVX: vpsrlw $2
}

define <16 x i16> @test4(<16 x i16> %a) {
  %div = udiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div

; AVX-LABEL: test4:
; AVX: vpmulhuw
; AVX: vpsubw
; AVX: vpsrlw $1
; AVX: vpaddw
; AVX: vpsrlw $2
; AVX-NOT: vpmulhuw
}

define <8 x i16> @test5(<8 x i16> %a) {
  %div = sdiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div

; SSE41-LABEL: test5:
; SSE41: pmulhw
; SSE41: psrlw $15
; SSE41: psraw $1
; SSE41: paddw

; AVX-LABEL: test5:
; AVX: vpmulhw
; AVX: vpsrlw $15
; AVX: vpsraw $1
; AVX: vpaddw
}

define <16 x i16> @test6(<16 x i16> %a) {
  %div = sdiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div

; AVX-LABEL: test6:
; AVX: vpmulhw
; AVX: vpsrlw $15
; AVX: vpsraw $1
; AVX: vpaddw
; AVX-NOT: vpmulhw
}

define <16 x i8> @test7(<16 x i8> %a) {
  %div = sdiv <16 x i8> %a, <i8 7, i8 7, i8 7, i8 7,i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7,i8 7, i8 7, i8 7, i8 7>
  ret <16 x i8> %div

; FIXME: scalarized
; SSE41-LABEL: test7:
; SSE41: pext
; AVX-LABEL: test7:
; AVX: pext
}

define <4 x i32> @test8(<4 x i32> %a) {
  %div = sdiv <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %div

; SSE41-LABEL: test8:
; SSE41: pmuldq
; SSE41: pshufd	$49
; SSE41: pshufd	$49
; SSE41: pmuldq
; SSE41: shufps	$-35
; SSE41: pshufd	$-40
; SSE41: padd
; SSE41: psrld $31
; SSE41: psrad $2
; SSE41: padd

; SSE-LABEL: test8:
; SSE: pmuludq
; SSE: pshufd	$49
; SSE: pshufd	$49
; SSE: pmuludq
; SSE: shufps	$-35
; SSE: pshufd	$-40
; SSE: psubd
; SSE: padd
; SSE: psrld $31
; SSE: psrad $2
; SSE: padd

; AVX-LABEL: test8:
; AVX: vpmuldq
; AVX: vpshufd	$49
; AVX: vpshufd	$49
; AVX: vpmuldq
; AVX: vshufps	$-35
; AVX: vpshufd	$-40
; AVX: vpadd
; AVX: vpsrld $31
; AVX: vpsrad $2
; AVX: vpadd
}

define <8 x i32> @test9(<8 x i32> %a) {
  %div = sdiv <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div

; AVX-LABEL: test9:
; AVX: vpbroadcastd
; AVX: vpalignr $4
; AVX: vpalignr $4
; AVX: vpmuldq
; AVX: vpmuldq
; AVX: vpalignr $4
; AVX: vpblendd $170
; AVX: vpadd
; AVX: vpsrld $31
; AVX: vpsrad $2
; AVX: vpadd
}

define <8 x i32> @test10(<8 x i32> %a) {
  %rem = urem <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %rem

; AVX-LABEL: test10:
; AVX: vpbroadcastd
; AVX: vpalignr $4
; AVX: vpmuludq
; AVX: vpmuludq
; AVX: vpblendd $170
; AVX: vpsubd
; AVX: vpsrld $1
; AVX: vpadd
; AVX: vpsrld $2
; AVX: vpmulld
}

define <8 x i32> @test11(<8 x i32> %a) {
  %rem = srem <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %rem

; AVX-LABEL: test11:
; AVX: vpbroadcastd
; AVX: vpalignr $4
; AVX: vpalignr $4
; AVX: vpmuldq
; AVX: vpmuldq
; AVX: vpalignr $4
; AVX: vpblendd $170
; AVX: vpadd
; AVX: vpsrld $31
; AVX: vpsrad $2
; AVX: vpadd
; AVX: vpmulld
}

define <2 x i16> @test12() {
  %I8 = insertelement <2 x i16> zeroinitializer, i16 -1, i32 0
  %I9 = insertelement <2 x i16> %I8, i16 -1, i32 1
  %B9 = urem <2 x i16> %I9, %I9
  ret <2 x i16> %B9

; AVX-LABEL: test12:
; AVX: xorps
}

define <4 x i32> @PR20355(<4 x i32> %a) {
; SSE-LABEL: PR20355:
; SSE:         movdqa {{(.*LCPI|__xmm@55555556555555565555555655555556).*}}, %[[X1:xmm[0-9]+]]
; SSE-NEXT:    movdqa %[[X1]], %[[X2:xmm[0-9]+]]
; SSE-NEXT:    psrad  $31, %[[X2]]
; SSE-NEXT:    pand   %xmm0, %[[X2]]
; SSE-NEXT:    movdqa %xmm0, %[[X3:xmm[0-9]+]]
; SSE-NEXT:    psrad  $31, %[[X3]]
; SSE-NEXT:    pand   %[[X1]], %[[X3]]
; SSE-NEXT:    paddd  %[[X2]], %[[X3]]
; SSE-NEXT:    pshufd {{.*}} #{{#?}} [[X4:xmm[0-9]+]] = xmm0[1,0,3,0]
; SSE-NEXT:    pmuludq %[[X1]], %xmm0
; SSE-NEXT:    pshufd {{.*}} #{{#?}} [[X1]] = [[X1]][1,0,3,0]
; SSE-NEXT:    pmuludq %[[X4]], %[[X1]]
; SSE-NEXT:    shufps {{.*}} #{{#?}} xmm0 = xmm0[1,3],[[X1]][1,3]
; SSE-NEXT:    pshufd {{.*}} #{{#?}} [[X5:xmm[0-9]+]] = xmm0[0,2,1,3]
; SSE-NEXT:    psubd  %[[X3]], %[[X5]]
; SSE-NEXT:    movdqa %[[X5]], %xmm0
; SSE-NEXT:    psrld  $31, %xmm0
; SSE-NEXT:    paddd  %[[X5]], %xmm0
; SSE-NEXT:    retq
;
; SSE41-LABEL: PR20355:
; SSE41:         movdqa {{(.*LCPI|__xmm@55555556555555565555555655555556).*}}, %[[X1:xmm[0-9]+]]
; SSE41-NEXT:    pshufd {{.*}} #{{#?}} [[X2:xmm[0-9]+]] = xmm0[1,0,3,0]
; SSE41-NEXT:    pmuldq %[[X1]], %xmm0
; SSE41-NEXT:    pshufd {{.*}} #{{#?}} [[X1]] = [[X1]][1,0,3,0]
; SSE41-NEXT:    pmuldq %[[X2]], %[[X1]]
; SSE41-NEXT:    shufps {{.*}} #{{#?}} xmm0 = xmm0[1,3],[[X1]][1,3]
; SSE41-NEXT:    pshufd {{.*}} #{{#?}} [[X3:xmm[0-9]+]] = xmm0[0,2,1,3]
; SSE41-NEXT:    movdqa %[[X3]], %xmm0
; SSE41-NEXT:    psrld  $31, %xmm0
; SSE41-NEXT:    paddd  %[[X3]], %xmm0
; SSE41-NEXT:    retq
entry:
  %sdiv = sdiv <4 x i32> %a, <i32 3, i32 3, i32 3, i32 3>
  ret <4 x i32> %sdiv
}
