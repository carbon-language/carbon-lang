; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=corei7 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=corei7-avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX


; Make sure that we generate non-temporal stores for the test cases below.

define void @test1(<4 x float>* %dst) {
; CHECK-LABEL: test1:
; SSE: movntps
; AVX: vmovntps
  store <4 x float> zeroinitializer, <4 x float>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test2(<4 x i32>* %dst) {
; CHECK-LABEL: test2:
; SSE: movntps
; AVX: vmovntps
  store <4 x i32> zeroinitializer, <4 x i32>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test3(<2 x double>* %dst) {
; CHECK-LABEL: test3:
; SSE: movntps
; AVX: vmovntps
  store <2 x double> zeroinitializer, <2 x double>* %dst, align 16, !nontemporal !1
  ret void
}

!1 = metadata !{i32 1}
