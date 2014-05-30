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

; Fold (vselect (build_vector AllOnes), N1, N2) -> N1

define <4 x float> @test14(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 true, i1 undef, i1 true, i1 undef>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test14
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: pcmpeq
; CHECK: ret

define <8 x i16> @test15(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 true, i1 true, i1 true, i1 undef, i1 undef, i1 true, i1 true, i1 undef>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test15
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: pcmpeq
; CHECK: ret

; Fold (vselect (build_vector AllZeros), N1, N2) -> N2

define <4 x float> @test16(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 false, i1 undef, i1 false, i1 undef>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
} 
; CHECK-LABEL: test16
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: ret 

define <8 x i16> @test17(<8 x i16> %a, <8 x i16> %b) {
  %1 = select <8 x i1> <i1 false, i1 false, i1 false, i1 undef, i1 undef, i1 false, i1 false, i1 undef>, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %1
}
; CHECK-LABEL: test17
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: ret

define <4 x float> @test18(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test18
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movss
; CHECK: ret

define <4 x i32> @test19(<4 x i32> %a, <4 x i32> %b) {
  %1 = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %1
}
; CHECK-LABEL: test19
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movss
; CHECK: ret

define <2 x double> @test20(<2 x double> %a, <2 x double> %b) {
  %1 = select <2 x i1> <i1 false, i1 true>, <2 x double> %a, <2 x double> %b
  ret <2 x double> %1
}
; CHECK-LABEL: test20
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK: ret

define <2 x i64> @test21(<2 x i64> %a, <2 x i64> %b) {
  %1 = select <2 x i1> <i1 false, i1 true>, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %1
}
; CHECK-LABEL: test21
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK: ret

define <4 x float> @test22(<4 x float> %a, <4 x float> %b) {
  %1 = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %1
}
; CHECK-LABEL: test22
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movss
; CHECK: ret

define <4 x i32> @test23(<4 x i32> %a, <4 x i32> %b) {
  %1 = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %1
}
; CHECK-LABEL: test23
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movss
; CHECK: ret

define <2 x double> @test24(<2 x double> %a, <2 x double> %b) {
  %1 = select <2 x i1> <i1 true, i1 false>, <2 x double> %a, <2 x double> %b
  ret <2 x double> %1
}
; CHECK-LABEL: test24
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK: ret

define <2 x i64> @test25(<2 x i64> %a, <2 x i64> %b) {
  %1 = select <2 x i1> <i1 true, i1 false>, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %1
}
; CHECK-LABEL: test25
; CHECK-NOT: psllw
; CHECK-NOT: psraw
; CHECK-NOT: xorps
; CHECK: movsd
; CHECK: ret

define <4 x float> @select_of_shuffles_0(<2 x float> %a0, <2 x float> %b0, <2 x float> %a1, <2 x float> %b1) {
; CHECK-LABEL: select_of_shuffles_0
; CHECK-DAG: movlhps %xmm2, [[REGA:%xmm[0-9]+]]
; CHECK-DAG: movlhps %xmm3, [[REGB:%xmm[0-9]+]]
; CHECK: subps [[REGB]], [[REGA]]
  %1 = shufflevector <2 x float> %a0, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %2 = shufflevector <2 x float> %a1, <2 x float> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %3 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x float> %2, <4 x float> %1
  %4 = shufflevector <2 x float> %b0, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %5 = shufflevector <2 x float> %b1, <2 x float> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %6 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x float> %5, <4 x float> %4
  %7 = fsub <4 x float> %3, %6
  ret <4 x float> %7
}
