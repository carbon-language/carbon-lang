; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define <4 x float> @foo(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: LCPI0_0
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-LABEL: foo:
; CHECK: cmpeqps %xmm1, %xmm0
; CHECK-NEXT: andps LCPI0_0(%rip), %xmm0
; CHECK-NEXT: retq

  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %result = sitofp <4 x i32> %ext to <4 x float>
  ret <4 x float> %result
}
