; RUN: llc < %s -march=x86 -mcpu=core2 | FileCheck %s
; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

; Verify that the backend correctly combines SSE2 builtin intrinsics.


define <4 x i32> @test_psra_1(<4 x i32> %A) {
  %1 = tail call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %A, i32 3)
  %2 = tail call <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32> %1, <4 x i32> <i32 3, i32 0, i32 7, i32 0>)
  %3 = tail call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %2, i32 2)
  ret <4 x i32> %3
}
; CHECK-LABEL: test_psra_1
; CHECK: psrad $8, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test_psra_2(<8 x i16> %A) {
  %1 = tail call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %A, i32 3)
  %2 = tail call <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16> %1, <8 x i16> <i16 3, i16 0, i16 0, i16 0, i16 7, i16 0, i16 0, i16 0>)
  %3 = tail call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %2, i32 2)
  ret <8 x i16> %3
}
; CHECK-LABEL: test_psra_2
; CHECK: psraw $8, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_psra_3(<4 x i32> %A) {
  %1 = tail call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %A, i32 0)
  %2 = tail call <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32> %1, <4 x i32> <i32 0, i32 0, i32 7, i32 0>)
  %3 = tail call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %2, i32 0)
  ret <4 x i32> %3
}
; CHECK-LABEL: test_psra_3
; CHECK-NOT: psrad
; CHECK: ret


define <8 x i16> @test_psra_4(<8 x i16> %A) {
  %1 = tail call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %A, i32 0)
  %2 = tail call <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16> %1, <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 7, i16 0, i16 0, i16 0>)
  %3 = tail call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %2, i32 0)
  ret <8 x i16> %3
}
; CHECK-LABEL: test_psra_4
; CHECK-NOT: psraw
; CHECK: ret


declare <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16>, i32)
declare <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32>, i32)

