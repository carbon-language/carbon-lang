; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Check that DAGCombiner correctly folds the following pairs of shuffles
; using the following rules:
;  1. shuffle(shuffle(x, y), undef) -> x
;  2. shuffle(shuffle(x, y), undef) -> y
;  3. shuffle(shuffle(x, y), undef) -> shuffle(x, undef)
;  4. shuffle(shuffle(x, y), undef) -> shuffle(undef, y)
;
; Rules 3. and 4. are used only if the resulting shuffle mask is legal.

define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 4, i32 3, i32 1>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 4, i32 0, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test1
; Mask: [3,0,0,1]
; CHECK: pshufd $67
; CHECK-NEXT: ret


define <4 x i32> @test2(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 4, i32 0, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test2
; Mask: [2,0,0,3]
; CHECK: pshufd $-62
; CHECK-NEXT: ret


define <4 x i32> @test3(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 4, i32 0, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test3
; Mask: [2,0,0,3]
; CHECK: pshufd $-62
; CHECK-NEXT: ret


define <4 x i32> @test4(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 4, i32 7, i32 1>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 4, i32 4, i32 0, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test4
; Mask: [0,0,0,1]
; CHECK: pshufd $64
; CHECK-NEXT: ret


define <4 x i32> @test5(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 5, i32 5, i32 2, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 4, i32 4, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: test5
; Mask: [1,1]
; CHECK: movhlps
; CHECK-NEXT: ret


define <4 x i32> @test6(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 6, i32 2, i32 4>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 4, i32 0, i32 4>
  ret <4 x i32> %2
}
; CHECK-LABEL: test6
; Mask: [2,0,0,0]
; CHECK: pshufd $2
; CHECK-NEXT: ret


define <4 x i32> @test7(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 0, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: test7
; Mask: [0,2,0,2]
; CHECK: pshufd $-120
; CHECK-NEXT: ret


define <4 x i32> @test8(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 4, i32 3, i32 4>
  ret <4 x i32> %2
}
; CHECK-LABEL: test8
; Mask: [1,0,3,0]
; CHECK: pshufd $49
; CHECK-NEXT: ret


define <4 x i32> @test9(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 1, i32 3, i32 2, i32 5>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 4, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: test9
; Mask: [1,3,0,2]
; CHECK: pshufd $-115
; CHECK-NEXT: ret


define <4 x i32> @test10(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 1, i32 1, i32 5, i32 5>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 0, i32 4, i32 1, i32 4>
  ret <4 x i32> %2
}
; CHECK-LABEL: test10
; Mask: [1,0,1,0]
; CHECK: pshufd $17
; CHECK-NEXT: ret


define <4 x i32> @test11(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 1, i32 2, i32 5, i32 4>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 0, i32 4, i32 1, i32 0>
  ret <4 x i32> %2
}
; CHECK-LABEL: test11
; Mask: [1,0,2,1]
; CHECK: pshufd $97
; CHECK-NEXT: ret


define <4 x i32> @test12(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 0, i32 2, i32 4>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 4, i32 0, i32 4>
  ret <4 x i32> %2
}
; CHECK-LABEL: test12
; Mask: [0,0,0,0]
; CHECK: pshufd $0
; CHECK-NEXT: ret


; The following pair of shuffles is folded into vector %A.
define <4 x i32> @test13(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 1, i32 4, i32 2, i32 6>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 4, i32 0, i32 2, i32 4>
  ret <4 x i32> %2
}
; CHECK-LABEL: test13
; CHECK-NOT: pshufd
; CHECK: ret


; The following pair of shuffles is folded into vector %B.
define <4 x i32> @test14(<4 x i32> %A, <4 x i32> %B) {
  %1 = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 0, i32 6, i32 2, i32 4>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 4, i32 1, i32 4>
  ret <4 x i32> %2
}
; CHECK-LABEL: test14
; CHECK-NOT: pshufd
; CHECK: ret

