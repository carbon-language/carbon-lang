; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s


; Verify that each of the following test cases is folded into a single
; instruction which performs a blend operation.

define <2 x i64> @test1(<2 x i64> %a, <2 x i64> %b) {
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 1>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}
; CHECK-LABEL: test1
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK-NOT: orps
; CHECK: ret


define <4 x i32> @test2(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test2
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK: ret


define <2 x i64> @test3(<2 x i64> %a, <2 x i64> %b) {
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 1>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}
; CHECK-LABEL: test3
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK-NEXT: ret


define <4 x i32> @test4(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 1, i32 2, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test4
; CHECK-NOT: xorps
; CHECK: movss
; CHECK-NOT: orps
; CHECK: ret


define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 1, i32 2, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test5
; CHECK-NOT: xorps
; CHECK: movss
; CHECK-NEXT: ret


define <4 x i32> @test6(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test6
; CHECK-NOT: xorps
; CHECK: blendps $12
; CHECK-NEXT: ret


define <4 x i32> @test7(<4 x i32> %a, <4 x i32> %b) {
  %and1 = and <4 x i32> %a, <i32 -1, i32 -1, i32 0, i32 0>
  %and2 = and <4 x i32> %b, <i32 0, i32 0, i32 -1, i32 -1>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}
; CHECK-LABEL: test7
; CHECK-NOT: xorps
; CHECK: blendps $12
; CHECK-NEXT: ret


define <2 x i64> @test8(<2 x i64> %a, <2 x i64> %b) {
  %and1 = and <2 x i64> %a, <i64 -1, i64 0>
  %and2 = and <2 x i64> %b, <i64 0, i64 -1>
  %or = or <2 x i64> %and1, %and2
  ret <2 x i64> %or
}
; CHECK-LABEL: test8
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK-NOT: orps
; CHECK: ret


define <4 x i32> @test9(<4 x i32> %a, <4 x i32> %b) {
  %and1 = and <4 x i32> %a, <i32 0, i32 0, i32 -1, i32 -1>
  %and2 = and <4 x i32> %b, <i32 -1, i32 -1, i32 0, i32 0>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}
; CHECK-LABEL: test9
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK: ret


define <2 x i64> @test10(<2 x i64> %a, <2 x i64> %b) {
  %and1 = and <2 x i64> %a, <i64 0, i64 -1>
  %and2 = and <2 x i64> %b, <i64 -1, i64 0>
  %or = or <2 x i64> %and1, %and2
  ret <2 x i64> %or
}
; CHECK-LABEL: test10
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK-NEXT: ret


define <4 x i32> @test11(<4 x i32> %a, <4 x i32> %b) {
  %and1 = and <4 x i32> %a, <i32 -1, i32 0, i32 0, i32 0>
  %and2 = and <4 x i32> %b, <i32 0, i32 -1, i32 -1, i32 -1>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}
; CHECK-LABEL: test11
; CHECK-NOT: xorps
; CHECK: movss
; CHECK-NOT: orps
; CHECK: ret


define <4 x i32> @test12(<4 x i32> %a, <4 x i32> %b) {
  %and1 = and <4 x i32> %a, <i32 0, i32 -1, i32 -1, i32 -1>
  %and2 = and <4 x i32> %b, <i32 -1, i32 0, i32 0, i32 0>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}
; CHECK-LABEL: test12
; CHECK-NOT: xorps
; CHECK: movss
; CHECK-NEXT: ret


; Verify that the following test cases are folded into single shuffles.

define <4 x i32> @test13(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 1, i32 1, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test13
; CHECK-NOT: xorps
; CHECK: shufps
; CHECK-NEXT: ret


define <2 x i64> @test14(<2 x i64> %a, <2 x i64> %b) {
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}
; CHECK-LABEL: test14
; CHECK-NOT: pslldq
; CHECK-NOT: por
; CHECK: punpcklqdq
; CHECK-NEXT: ret


define <4 x i32> @test15(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 1>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 2, i32 1, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test15
; CHECK-NOT: xorps
; CHECK: shufps
; CHECK-NOT: shufps
; CHECK-NOT: orps
; CHECK: ret


define <2 x i64> @test16(<2 x i64> %a, <2 x i64> %b) {
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}
; CHECK-LABEL: test16
; CHECK-NOT: pslldq
; CHECK-NOT: por
; CHECK: punpcklqdq
; CHECK: ret


; Verify that the dag-combiner does not fold a OR of two shuffles into a single
; shuffle instruction when the shuffle indexes are not compatible.

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 0, i32 4, i32 2>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test17
; CHECK: por
; CHECK-NEXT: ret


define <4 x i32> @test18(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 0, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test18
; CHECK: orps
; CHECK: ret


define <4 x i32> @test19(<4 x i32> %a, <4 x i32> %b) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 0, i32 4, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 2, i32 2>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test19
; CHECK: por
; CHECK-NEXT: ret


define <2 x i64> @test20(<2 x i64> %a, <2 x i64> %b) {
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}
; CHECK-LABEL: test20
; CHECK-NOT: xorps
; CHECK: orps
; CHECK-NEXT: movq
; CHECK-NEXT: ret


define <2 x i64> @test21(<2 x i64> %a, <2 x i64> %b) {
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}
; CHECK-LABEL: test21
; CHECK: por
; CHECK-NEXT: pslldq
; CHECK-NEXT: ret

; Verify that the DAGCombiner doesn't crash in the attempt to check if a shuffle
; with illegal type has a legal mask. Method 'isShuffleMaskLegal' only knows how to
; handle legal vector value types.
define <4 x i8> @test_crash(<4 x i8> %a, <4 x i8> %b) {
  %shuf1 = shufflevector <4 x i8> %a, <4 x i8> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %shuf2 = shufflevector <4 x i8> %b, <4 x i8> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %or = or <4 x i8> %shuf1, %shuf2
  ret <4 x i8> %or
}
; CHECK-LABEL: test_crash
; CHECK: movsd
; CHECK: ret

