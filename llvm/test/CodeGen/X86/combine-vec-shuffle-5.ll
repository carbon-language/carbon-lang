; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Verify that the DAGCombiner correctly folds all the shufflevector pairs
; into a single shuffle operation.

define <4 x float> @test1(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test1
; Mask: [0,1,2,3]
; CHECK: movaps
; CHECK: ret

define <4 x float> @test2(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test2
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x float> @test3(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 4, i32 6, i32 0, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test3
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x float> @test4(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test4
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x float> @test5(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %b, <4 x float> %1, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: test5
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret


define <4 x i32> @test6(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i32> %b, <4 x i32> %1, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test6
; Mask: [4,5,6,7]
; CHECK: movaps
; CHECK: ret

define <4 x i32> @test7(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %2 = shufflevector <4 x i32> %b, <4 x i32> %1, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test7
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x i32> @test8(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x i32> %b, <4 x i32> %1, <4 x i32> <i32 4, i32 6, i32 0, i32 5>
  ret <4 x i32> %2
}
; CHECK-LABEL: test8
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x i32> @test9(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 3, i32 5, i32 5>
  %2 = shufflevector <4 x i32> %b, <4 x i32> %1, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x i32> %2
}
; CHECK-LABEL: test9
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK-NEXT: ret

define <4 x i32> @test10(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i32> %b, <4 x i32> %1, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test10
; Mask: [4,1,6,7]
; CHECK: blendps
; CHECK: ret

define <4 x float> @test11(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test11
; Mask: [0,1,2,3]
; CHECK-NOT: movaps
; CHECK-NOT: blendps
; CHECK: ret

define <4 x float> @test12(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test12
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x float> @test13(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test13
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x float> @test14(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 6, i32 7, i32 5, i32 5>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x float> %2
}
; CHECK-LABEL: test14
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK: ret

define <4 x float> @test15(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  %2 = shufflevector <4 x float> %a, <4 x float> %1, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x float> %2
}
; CHECK-LABEL: test15
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret

define <4 x i32> @test16(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i32> %a, <4 x i32> %1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test16
; Mask: [0,1,2,3]
; CHECK-NOT: movaps
; CHECK-NOT: blendps
; CHECK: ret

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  %2 = shufflevector <4 x i32> %a, <4 x i32> %1, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test17
; Mask: [0,5,6,7]
; CHECK: movss
; CHECK: ret

define <4 x i32> @test18(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %2 = shufflevector <4 x i32> %a, <4 x i32> %1, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test18
; Mask: [0,1,4,5]
; CHECK: movlhps
; CHECK: ret

define <4 x i32> @test19(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 6, i32 7, i32 5, i32 5>
  %2 = shufflevector <4 x i32> %a, <4 x i32> %1, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x i32> %2
}
; CHECK-LABEL: test19
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK: ret

define <4 x i32> @test20(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  %2 = shufflevector <4 x i32> %a, <4 x i32> %1, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test20
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret

; Verify that we correctly fold shuffles even when we use illegal vector types.
define <4 x i8> @test1c(<4 x i8>* %a, <4 x i8>* %b) {
  %A = load <4 x i8>* %a
  %B = load <4 x i8>* %b
  %1 = shufflevector <4 x i8> %A, <4 x i8> %B, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %2 = shufflevector <4 x i8> %B, <4 x i8> %1, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
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
  %2 = shufflevector <4 x i8> %B, <4 x i8> %1, <4 x i32> <i32 4, i32 6, i32 0, i32 5>
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
  %2 = shufflevector <4 x i8> %B, <4 x i8> %1, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x i8> %2
}
; CHECK-LABEL: test3c
; Mask: [6,7,2,3]
; CHECK: movhlps
; CHECK: ret

define <4 x i8> @test4c(<4 x i8>* %a, <4 x i8>* %b) {
  %A = load <4 x i8>* %a
  %B = load <4 x i8>* %b
  %1 = shufflevector <4 x i8> %A, <4 x i8> %B, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i8> %B, <4 x i8> %1, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ret <4 x i8> %2
}
; CHECK-LABEL: test4c
; Mask: [4,1,6,7]
; CHECK: blendps $13
; CHECK: ret

