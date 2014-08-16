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


; Verify that the dag combiner correctly folds the following shuffle pairs to Undef.

define <4 x i32> @test1b(<4 x i32> %A) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 5, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test1b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test2b(<4 x i32> %A) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> <i32 4, i32 5, i32 1, i32 6>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 6, i32 7>
  ret <4 x i32> %2
}
; CHECK-LABEL: test2b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test3b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %B, <4 x i32> <i32 1, i32 3, i32 1, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test3b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test4b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> <i32 4, i32 1, i32 1, i32 6>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %B, <4 x i32> <i32 0, i32 3, i32 3, i32 0>
  ret <4 x i32> %2
}
; CHECK-LABEL: test4b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test5b(<4 x i32> %A) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 5, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  ret <4 x i32> %3
}
; CHECK-LABEL: test5b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test6b(<4 x i32> %A) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> <i32 4, i32 5, i32 1, i32 6>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 6, i32 7>
  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i32> %3
}
; CHECK-LABEL: test6b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test7b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %B, <4 x i32> <i32 1, i32 3, i32 1, i32 3>
  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 6>
  ret <4 x i32> %3
}
; CHECK-LABEL: test7b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test8b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 4, i32 1, i32 1, i32 6>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 0, i32 4, i32 1, i32 6>
  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> <i32 1, i32 5, i32 3, i32 3>
  ret <4 x i32> %3
}
; CHECK-LABEL: test8b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test9b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 1, i32 undef, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 4, i32 2, i32 1>
  %3 = shufflevector <4 x i32> %2, <4 x i32> %A, <4 x i32> <i32 2, i32 1, i32 1, i32 2>
  ret <4 x i32> %3
}
; CHECK-LABEL: test9b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test10b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 undef, i32 undef, i32 1, i32 6>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %A, <4 x i32> <i32 0, i32 6, i32 1, i32 0>
  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 0, i32 2>
  ret <4 x i32> %3
}
; CHECK-LABEL: test10b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test11b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 undef, i32 1, i32 undef>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %B, <4 x i32> <i32 1, i32 3, i32 1, i32 3>
  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> <i32 0, i32 5, i32 1, i32 6>
  ret <4 x i32> %3
}
; CHECK-LABEL: test11b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <4 x i32> @test12b(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 undef, i32 1, i32 1, i32 undef>
  %2 = shufflevector <4 x i32> %1, <4 x i32> %B, <4 x i32> <i32 0, i32 3, i32 3, i32 0>
  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> <i32 1, i32 5, i32 1, i32 4>
  ret <4 x i32> %3
}
; CHECK-LABEL: test12b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <8 x i32> @test13b(<8 x i32> %A, <8 x i32> %B) {
  %1 = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 0, i32 undef, i32 1, i32 undef, i32 0, i32 undef, i32 1, i32 undef>
  %2 = shufflevector <8 x i32> %1, <8 x i32> %B, <8 x i32> <i32 1, i32 3, i32 1, i32 3, i32 1, i32 3, i32 1, i32 3>
  %3 = shufflevector <8 x i32> %2, <8 x i32> undef, <8 x i32> <i32 0, i32 9, i32 1, i32 10, i32 0, i32 9, i32 1, i32 10>
  ret <8 x i32> %3
}
; CHECK-LABEL: test13b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <8 x i32> @test14b(<8 x i32> %A, <8 x i32> %B) {
  %1 = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 undef, i32 1, i32 1, i32 undef, i32 undef, i32 1, i32 1, i32 undef>
  %2 = shufflevector <8 x i32> %1, <8 x i32> %B, <8 x i32> <i32 0, i32 3, i32 3, i32 0, i32 0, i32 3, i32 3, i32 0>
  %3 = shufflevector <8 x i32> %2, <8 x i32> undef, <8 x i32> <i32 1, i32 9, i32 1, i32 8, i32 1, i32 9, i32 1, i32 8>
  ret <8 x i32> %3
}
; CHECK-LABEL: test14b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <8 x i32> @test15b(<8 x i32> %A, <8 x i32> %B) {
  %1 = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 0, i32 1, i32 undef, i32 11, i32 0, i32 1, i32 undef, i32 11>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 8, i32 9, i32 2, i32 11, i32 8, i32 9, i32 2, i32 11>
  %3 = shufflevector <8 x i32> %2, <8 x i32> %A, <8 x i32> <i32 2, i32 2, i32 undef, i32 2, i32 2, i32 2, i32 undef, i32 2>
  ret <8 x i32> %3
}
; CHECK-LABEL: test15b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

define <8 x i32> @test16b(<8 x i32> %A, <8 x i32> %B) {
  %1 = shufflevector <8 x i32> %A, <8 x i32> %B, <8 x i32> <i32 undef, i32 undef, i32 1, i32 10, i32 undef, i32 undef, i32 1, i32 10>
  %2 = shufflevector <8 x i32> %1, <8 x i32> %A, <8 x i32> <i32 0, i32 10, i32 2, i32 11, i32 0, i32 10, i32 2, i32 11>
  %3 = shufflevector <8 x i32> %2, <8 x i32> undef, <8 x i32> <i32 4, i32 9, i32 undef, i32 0, i32 4, i32 9, i32 undef, i32 0>
  ret <8 x i32> %3
}
; CHECK-LABEL: test16b
; CHECK-NOT: blendps
; CHECK-NOT: pshufd
; CHECK-NOT: movhlps
; CHECK: ret

