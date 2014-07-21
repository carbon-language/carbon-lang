; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Verify that we fold shuffles according to rule:
;  (shuffle(shuffle A, Undef, M0), B, M1) -> (shuffle A, B, M2)

define <4 x float> @test1(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 4, i32 2, i32 3, i32 1>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 4, i32 5, i32 1, i32 2>
  ret <4 x float> %2
}
; CHECK-LABEL: test1
; Mask: [4,5,2,3]
; CHECK: movsd
; CHECK: ret

define <4 x float> @test2(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 6, i32 0, i32 1, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 1, i32 2, i32 4, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test2
; Mask: [0,1,4,5]
; CHECK: movlhps 
; CHECK: ret

define <4 x float> @test3(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test3
; Mask: [0,1,4,u]
; CHECK: movlhps
; CHECK: ret

define <4 x float> @test4(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test4
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x float> @test5(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 4, i32 1, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 6, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test5
; Mask: [0,1,6,7]
; CHECK: blendps $12
; CHECK: ret

; Verify that we fold shuffles according to rule:
;  (shuffle(shuffle A, Undef, M0), A, M1) -> (shuffle A, Undef, M2)

define <4 x float> @test6(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 4, i32 2, i32 3, i32 1>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 1, i32 2>
  ret <4 x float> %2
}
; CHECK-LABEL: test6
; Mask: [0,1,2,3]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK-NOT: movlhps
; CHECK: ret

define <4 x float> @test7(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 6, i32 0, i32 1, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 1, i32 2, i32 4, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test7
; Mask: [0,1,0,1]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK: movlhps 
; CHECK-NEXT: ret

define <4 x float> @test8(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 0, i32 2, i32 4, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test8
; Mask: [0,1,0,u]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK: movlhps
; CHECK-NEXT: ret

define <4 x float> @test9(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test9
; Mask: [2,3,2,3]
; CHECK-NOT: movlhps
; CHECK-NOT: palignr
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x float> @test10(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 4, i32 1, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> %a, <4 x i32> <i32 0, i32 2, i32 6, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test10
; Mask: [0,1,2,3]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK-NOT: movlhps
; CHECK: ret

define <4 x float> @test11(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 4, i32 2, i32 3, i32 1>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 0, i32 1, i32 5, i32 6>
  ret <4 x float> %2
}
; CHECK-LABEL: test11
; Mask: [4,5,2,3]
; CHECK: movsd
; CHECK: ret

define <4 x float> @test12(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 6, i32 0, i32 1, i32 7>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 5, i32 6, i32 0, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test12
; Mask: [0,1,4,5]
; CHECK: movlhps 
; CHECK: ret

define <4 x float> @test13(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 4, i32 5, i32 0, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test13
; Mask: [0,1,4,u]
; CHECK: movlhps
; CHECK: ret

define <4 x float> @test14(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test14
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x float> @test15(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 4, i32 1, i32 3>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 4, i32 6, i32 2, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test15
; Mask: [0,1,6,7]
; CHECK: blendps $12
; CHECK: ret

; Verify that shuffles are canonicalized according to rules:
;  shuffle(B, shuffle(A, Undef)) -> shuffle(shuffle(A, Undef), B)
;
; This allows to trigger the following combine rule:
;  (shuffle(shuffle A, Undef, M0), A, M1) -> (shuffle A, Undef, M2)
;
; As a result, all the shuffle pairs in each function below should be
; combined into a single legal shuffle operation.

define <4 x float> @test16(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 4, i32 2, i32 3, i32 1>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test16
; Mask: [0,1,2,3]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK-NOT: movlhps
; CHECK: ret

define <4 x float> @test17(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 6, i32 0, i32 1, i32 7>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 5, i32 6, i32 0, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: test17
; Mask: [0,1,0,1]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK: movlhps 
; CHECK-NEXT: ret

define <4 x float> @test18(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 4, i32 6, i32 0, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test18
; Mask: [0,1,0,u]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK: movlhps
; CHECK-NEXT: ret

define <4 x float> @test19(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test19
; Mask: [2,3,2,3]
; CHECK-NOT: movlhps
; CHECK-NOT: palignr
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x float> @test20(<4 x float> %a) {
  %1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 0, i32 4, i32 1, i32 3>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 4, i32 6, i32 2, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test20
; Mask: [0,1,2,3]
; CHECK-NOT: pshufd
; CHECK-NOT: shufps
; CHECK-NOT: movlhps
; CHECK: ret

