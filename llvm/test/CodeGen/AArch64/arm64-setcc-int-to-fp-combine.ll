; RUN: llc < %s -asm-verbose=false -mtriple=arm64-apple-ios | FileCheck %s

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
; Make sure the operation doesn't try to get folded when the sizes don't match,
; as that ends up crashing later when trying to form a bitcast operation for
; the folded nodes.
define void @foo1(<4 x float> %val, <4 x float> %test, <4 x double>* %p) nounwind {
; CHECK-LABEL: foo1:
; CHECK: movi.4s
; CHECK: scvtf.2d
; CHECK: scvtf.2d
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %result = sitofp <4 x i32> %ext to <4 x double>
  store <4 x double> %result, <4 x double>* %p
  ret void
}

; Fold explicit AND operations when the constant isn't a splat of a single
; scalar value like what the zext creates.
define <4 x float> @foo2(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: lCPI2_0:
; CHECK-NEXT: .long 1065353216
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long 1065353216
; CHECK-NEXT: .long 0
; CHECK-LABEL: foo2:
; CHECK: adrp  x8, lCPI2_0@PAGE
; CHECK: ldr q2, [x8, lCPI2_0@PAGEOFF]
; CHECK-NEXT:  fcmeq.4s  v0, v0, v1
; CHECK-NEXT:  and.16b v0, v0, v2
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %and = and <4 x i32> %ext, <i32 255, i32 256, i32 257, i32 258>
  %result = sitofp <4 x i32> %and to <4 x float>
  ret <4 x float> %result
}
