; RUN: llc < %s -asm-verbose=false -march=arm64 -aarch64-neon-syntax=apple | FileCheck %s

define <4 x float> @foo(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: foo:
; CHECK-NEXT: fcmeq.4s  v0, v0, v1
; CHECK-NEXT: fmov.4s v1, #1.00000000
; CHECK-NEXT: and.16b v0, v0, v1
; CHECK-NEXT: ret
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %result = sitofp <4 x i32> %ext to <4 x float>
  ret <4 x float> %result
}
