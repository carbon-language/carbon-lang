; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2


; Verify that the following shifts are lowered into a sequence of two shifts plus
; a blend. On pre-avx2 targets, instead of scalarizing logical and arithmetic
; packed shift right by a constant build_vector the backend should always try to
; emit a simpler sequence of two shifts + blend when possible.

define <8 x i16> @test1(<8 x i16> %a) {
  %lshr = lshr <8 x i16> %a, <i16 3, i16 3, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <8 x i16> %lshr
}
; CHECK-LABEL: test1
; SSE: psrlw
; SSE-NEXT: psrlw
; SSE-NEXT: movss
; AVX: vpsrlw
; AVX-NEXT: vpsrlw
; AVX-NEXT: vmovss
; AVX2: vpsrlw
; AVX2-NEXT: vpsrlw
; AVX2-NEXT: vmovss
; CHECK: ret


define <8 x i16> @test2(<8 x i16> %a) {
  %lshr = lshr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 2, i16 2, i16 2, i16 2>
  ret <8 x i16> %lshr
}
; CHECK-LABEL: test2
; SSE: psrlw
; SSE-NEXT: psrlw
; SSE-NEXT: movsd
; AVX: vpsrlw
; AVX-NEXT: vpsrlw
; AVX-NEXT: vmovsd
; AVX2: vpsrlw
; AVX2-NEXT: vpsrlw
; AVX2-NEXT: vmovsd
; CHECK: ret


define <4 x i32> @test3(<4 x i32> %a) {
  %lshr = lshr <4 x i32> %a, <i32 3, i32 2, i32 2, i32 2>
  ret <4 x i32> %lshr
}
; CHECK-LABEL: test3
; SSE: psrld
; SSE-NEXT: psrld
; SSE-NEXT: movss
; AVX: vpsrld
; AVX-NEXT: vpsrld
; AVX-NEXT: vmovss
; AVX2: vpsrlvd
; CHECK: ret


define <4 x i32> @test4(<4 x i32> %a) {
  %lshr = lshr <4 x i32> %a, <i32 3, i32 3, i32 2, i32 2>
  ret <4 x i32> %lshr
}
; CHECK-LABEL: test4
; SSE: psrld
; SSE-NEXT: psrld
; SSE-NEXT: movsd
; AVX: vpsrld
; AVX-NEXT: vpsrld
; AVX-NEXT: vmovsd
; AVX2: vpsrlvd
; CHECK: ret


define <8 x i16> @test5(<8 x i16> %a) {
  %lshr = ashr <8 x i16> %a, <i16 3, i16 3, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <8 x i16> %lshr
}

define <8 x i16> @test6(<8 x i16> %a) {
  %lshr = ashr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 2, i16 2, i16 2, i16 2>
  ret <8 x i16> %lshr
}
; CHECK-LABEL: test6
; SSE: psraw
; SSE-NEXT: psraw
; SSE-NEXT: movsd
; AVX: vpsraw
; AVX-NEXT: vpsraw
; AVX-NEXT: vmovsd
; AVX2: vpsraw
; AVX2-NEXT: vpsraw
; AVX2-NEXT: vmovsd
; CHECK: ret


define <4 x i32> @test7(<4 x i32> %a) {
  %lshr = ashr <4 x i32> %a, <i32 3, i32 2, i32 2, i32 2>
  ret <4 x i32> %lshr
}
; CHECK-LABEL: test7
; SSE: psrad
; SSE-NEXT: psrad
; SSE-NEXT: movss
; AVX: vpsrad
; AVX-NEXT: vpsrad
; AVX-NEXT: vmovss
; AVX2: vpsravd
; CHECK: ret


define <4 x i32> @test8(<4 x i32> %a) {
  %lshr = ashr <4 x i32> %a, <i32 3, i32 3, i32 2, i32 2>
  ret <4 x i32> %lshr
}
; CHECK-LABEL: test8
; SSE: psrad
; SSE-NEXT: psrad
; SSE-NEXT: movsd
; AVX: vpsrad
; AVX-NEXT: vpsrad
; AVX-NEXT: vmovsd
; AVX2: vpsravd
; CHECK: ret

