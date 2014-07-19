; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

define <4 x float> @test1(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test1
; Mask: [0,1,2,3]
; CHECK: movaps
; CHECK: ret

define <4 x float> @test2(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test2
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x float> @test3(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test3
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x float> @test4(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test4
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x float> @test5(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test5
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret


define <4 x i32> @test6(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test6
; Mask: [4,5,6,7]
; CHECK: movaps
; CHECK: ret

define <4 x i32> @test7(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test7
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x i32> @test8(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 1>
  ret <4 x i32> %2
}
; CHECK-LABEL: test8
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x i32> @test9(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i32> %2
}
; CHECK-LABEL: test9
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x i32> @test10(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test10
; Mask: [4,1,6,7]
; CHECK: blendps
; CHECK: ret

define <4 x float> @test11(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test11
; Mask: [0,1,2,3]
; CHECK-NOT: movaps
; CHECK-NOT: blendps
; CHECK: ret

define <4 x float> @test12(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test12
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x float> @test13(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test13
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x float> @test14(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 6, i32 7, i32 5, i32 5>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test14
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK: ret

define <4 x float> @test15(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test15
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret

define <4 x i32> @test16(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %a, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test16
; Mask: [0,1,2,3]
; CHECK-NOT: movaps
; CHECK-NOT: blendps
; CHECK: ret

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %a, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test17
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x i32> @test18(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test18
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x i32> @test19(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 6, i32 7, i32 5, i32 5>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %a, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i32> %2
}
; CHECK-LABEL: test19
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK: ret

define <4 x i32> @test20(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %a, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test20
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret

; Check some negative cases.
define <4 x float> @test1b(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 0>
  ret <4 x float> %2
}
; CHECK-LABEL: test1b
; CHECK: shufps
; CHECK: shufps
; CHECK: ret

define <4 x float> @test2b(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 0, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test2b
; CHECK: shufps
; CHECK: pshufd
; CHECK: ret

define <4 x float> @test3b(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 6, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 7, i32 2, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test3b
; CHECK: shufps
; CHECK: shufps
; CHECK: ret

define <4 x float> @test4b(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 5, i32 5, i32 2, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test4b
; CHECK: shufps
; CHECK: shufps
; CHECK: ret


; Verify that we correctly fold shuffles even when we use illegal vector types.
define <4 x i8> @test1c(<4 x i8>* %a, <4 x i8>* %b) {
  %A = load <4 x i8>* %a
  %B = load <4 x i8>* %b
  %1 = shufflevector <4 x i8> %A, <4 x i8> %B, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %2 = shufflevector <4 x i8> %1, <4 x i8> %B, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x i8> %2
}
; CHECK-LABEL: test1c
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK-NEXT: ret

define <4 x i8> @test2c(<4 x i8>* %a, <4 x i8>* %b) {
  %A = load <4 x i8>* %a
  %B = load <4 x i8>* %b
  %1 = shufflevector <4 x i8> %A, <4 x i8> %B, <4 x i32> <i32 0, i32 5, i32 1, i32 5>
  %2 = shufflevector <4 x i8> %1, <4 x i8> %B, <4 x i32> <i32 0, i32 2, i32 4, i32 1>
  ret <4 x i8> %2
}
; CHECK-LABEL: test2c
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK-NEXT: ret

define <4 x i8> @test3c(<4 x i8>* %a, <4 x i8>* %b) {
  %A = load <4 x i8>* %a
  %B = load <4 x i8>* %b
  %1 = shufflevector <4 x i8> %A, <4 x i8> %B, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x i8> %1, <4 x i8> %B, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i8> %2
}
; CHECK-LABEL: test3c
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x i8> @test4c(<4 x i8>* %a, <4 x i8>* %b) {
  %A = load <4 x i8>* %a
  %B = load <4 x i8>* %b
  %1 = shufflevector <4 x i8> %A, <4 x i8> %B, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i8> %1, <4 x i8> %B, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x i8> %2
}
; CHECK-LABEL: test4c
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret

; The following test cases are generated from this C++ code
; 
;__m128 blend_01(__m128 a, __m128 b)
;{
;  __m128 s = a;
;  s = _mm_blend_ps( s, b, 1<<0 );
;  s = _mm_blend_ps( s, b, 1<<1 );
;  return s;
;}
;
;__m128 blend_02(__m128 a, __m128 b)
;{
;  __m128 s = a;
;  s = _mm_blend_ps( s, b, 1<<0 );
;  s = _mm_blend_ps( s, b, 1<<2 );
;  return s;
;}
;
;__m128 blend_123(__m128 a, __m128 b)
;{
;  __m128 s = a;
;  s = _mm_blend_ps( s, b, 1<<1 );
;  s = _mm_blend_ps( s, b, 1<<2 );
;  s = _mm_blend_ps( s, b, 1<<3 );
;  return s;
;}

; Ideally, we should collapse the following shuffles into a single one.

define <4 x float> @blend_01(<4 x float> %a, <4 x float> %b) {
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 undef, i32 2, i32 3>
  %shuffle6 = shufflevector <4 x float> %shuffle, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x float> %shuffle6
}
; CHECK-LABEL: blend_01
; CHECK: movsd
; CHECK-NEXT: ret

define <4 x float> @blend_02(<4 x float> %a, <4 x float> %b) {
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 undef, i32 3>
  %shuffle6 = shufflevector <4 x float> %shuffle, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x float> %shuffle6
}
; CHECK-LABEL: blend_02
; CHECK: blendps $5
; CHECK-NEXT: ret

define <4 x float> @blend_123(<4 x float> %a, <4 x float> %b) {
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 undef, i32 undef>
  %shuffle6 = shufflevector <4 x float> %shuffle, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 undef>
  %shuffle12 = shufflevector <4 x float> %shuffle6, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %shuffle12
}
; CHECK-LABEL: blend_123
; CHECK: movss
; CHECK: ret

define <4 x i32> @test_movhl_1(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 7, i32 5, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 6, i32 1, i32 0, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test_movhl_1
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x i32> @test_movhl_2(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 0, i32 3, i32 6>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 3, i32 7, i32 0, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: test_movhl_2
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x i32> @test_movhl_3(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 7, i32 6, i32 3, i32 2>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %b, <4 x i32> <i32 6, i32 0, i32 3, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: test_movhl_3
; CHECK: movhlps
; CHECK-NEXT: ret

