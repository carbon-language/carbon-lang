; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=corei7 -mattr=-sse4.1 < %s | FileCheck %s

; Verify that we don't emit packed vector shifts instructions if the
; condition used by the vector select is a vector of constants.


define <4 x float> @test1(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test1
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret


define <4 x float> @test2(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test2
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret


define <4 x float> @test3(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test3
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret


define <4 x float> @test4(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 false, i1 false, i1 false, i1 false>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test4
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: movaps  %xmm1, %xmm0
; CHECK: ret


define <4 x float> @test5(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test5
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret


define <8 x i16> @test6(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x i16> %a, <8 x i16> %a
  ret <8 x i16> %1
}
; CHECK-LABEL: test6
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret


define <8 x i16> @test7(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test7
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret


define <8 x i16> @test8(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test8
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret

define <8 x i16> @test9(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test9
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: movaps  %xmm1, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test10(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test10
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret

define <8 x i16> @test11(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 false, i1 true, i1 true, i1 false, i1 undef, i1 true, i1 true, i1 undef>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test11
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret

define <8 x i16> @test12(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 false, i1 false, i1 undef, i1 false, i1 false, i1 false, i1 false, i1 undef>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test12
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret

define <8 x i16> @test13(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 undef, i1 undef, i1 undef, i1 undef, i1 undef, i1 undef, i1 undef, i1 undef>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test13
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK: ret


