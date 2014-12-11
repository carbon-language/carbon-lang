; RUN: llc < %s -x86-experimental-vector-shuffle-lowering=false -mattr=+avx2 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

; PR21876
; The old shuffle lowering sometimes generates VZEXT nodes with both input
; and output same-sized types, here 256-bits.  For instance, a v8i8 to v8i32
; zero-extend would become a (v8i32 (VZEXT v32i8)) node, which can't happen
; otherwise.  The companion commit r223996 added those patterns temporarily.
; This test, along with the VR256 for AVX2 PMOVXrr instructions, should be
; removed once the old vector shuffle lowering goes away.

define void @test_avx2_pmovx_256(<8 x i8>* %tmp64, <8 x float>* %tmp75) {
; CHECK-LABEL: test_avx2_pmovx_256
; We really don't care about the generated code.
; CHECK: vpmovzxbd
; CHECK: vpbroadcastd
; CHECK: vpand
; CHECK: vcvtdq2ps
; CHECK: vmovups
; CHECK: vzeroupper
; CHECK: retq

  %wide.load458 = load <8 x i8>* %tmp64, align 1
  %tmp68 = uitofp <8 x i8> %wide.load458 to <8 x float>
  store <8 x float> %tmp68, <8 x float>* %tmp75, align 4
  ret void
}
