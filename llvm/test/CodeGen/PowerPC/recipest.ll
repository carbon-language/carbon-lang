; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-unsafe-fp-math | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck -check-prefix=CHECK-SAFE %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare double @llvm.sqrt.f64(double)
declare float @llvm.sqrt.f32(float)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define double @foo(double %a, double %b) nounwind {
entry:
  %x = call double @llvm.sqrt.f64(double %b)
  %r = fdiv double %a, %x
  ret double %r

; CHECK: @foo
; CHECK: frsqrte
; CHECK: fnmsub
; CHECK: fmul
; CHECK: fmadd
; CHECK: fmul
; CHECK: fmul
; CHECK: fmadd
; CHECK: fmul
; CHECK: fmul
; CHECK: blr

; CHECK-SAFE: @foo
; CHECK-SAFE: fsqrt
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define float @goo(float %a, float %b) nounwind {
entry:
  %x = call float @llvm.sqrt.f32(float %b)
  %r = fdiv float %a, %x
  ret float %r

; CHECK: @goo
; CHECK: frsqrtes
; CHECK: fnmsubs
; CHECK: fmuls
; CHECK: fmadds
; CHECK: fmuls
; CHECK: fmuls
; CHECK: blr

; CHECK-SAFE: @goo
; CHECK-SAFE: fsqrts
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define <4 x float> @hoo(<4 x float> %a, <4 x float> %b) nounwind {
entry:
  %x = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %r = fdiv <4 x float> %a, %x
  ret <4 x float> %r

; CHECK: @hoo
; CHECK: vrsqrtefp

; CHECK-SAFE: @hoo
; CHECK-SAFE-NOT: vrsqrtefp
; CHECK-SAFE: blr
}

define double @foo2(double %a, double %b) nounwind {
entry:
  %r = fdiv double %a, %b
  ret double %r

; CHECK: @foo2
; CHECK: fre
; CHECK: fnmsub
; CHECK: fmadd
; CHECK: fnmsub
; CHECK: fmadd
; CHECK: fmul
; CHECK: blr

; CHECK-SAFE: @foo2
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define float @goo2(float %a, float %b) nounwind {
entry:
  %r = fdiv float %a, %b
  ret float %r

; CHECK: @goo2
; CHECK: fres
; CHECK: fnmsubs
; CHECK: fmadds
; CHECK: fmuls
; CHECK: blr

; CHECK-SAFE: @goo2
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define <4 x float> @hoo2(<4 x float> %a, <4 x float> %b) nounwind {
entry:
  %r = fdiv <4 x float> %a, %b
  ret <4 x float> %r

; CHECK: @hoo2
; CHECK: vrefp

; CHECK-SAFE: @hoo2
; CHECK-SAFE-NOT: vrefp
; CHECK-SAFE: blr
}

define double @foo3(double %a) nounwind {
entry:
  %r = call double @llvm.sqrt.f64(double %a)
  ret double %r

; CHECK: @foo3
; CHECK: frsqrte
; CHECK: fnmsub
; CHECK: fmul
; CHECK: fmadd
; CHECK: fmul
; CHECK: fmul
; CHECK: fmadd
; CHECK: fmul
; CHECK: fre
; CHECK: fnmsub
; CHECK: fmadd
; CHECK: fnmsub
; CHECK: fmadd
; CHECK: blr

; CHECK-SAFE: @foo3
; CHECK-SAFE: fsqrt
; CHECK-SAFE: blr
}

define float @goo3(float %a) nounwind {
entry:
  %r = call float @llvm.sqrt.f32(float %a)
  ret float %r

; CHECK: @goo3
; CHECK: frsqrtes
; CHECK: fnmsubs
; CHECK: fmuls
; CHECK: fmadds
; CHECK: fmuls
; CHECK: fres
; CHECK: fnmsubs
; CHECK: fmadds
; CHECK: blr

; CHECK-SAFE: @goo3
; CHECK-SAFE: fsqrts
; CHECK-SAFE: blr
}

define <4 x float> @hoo3(<4 x float> %a) nounwind {
entry:
  %r = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %r

; CHECK: @hoo3
; CHECK: vrsqrtefp
; CHECK: vrefp

; CHECK-SAFE: @hoo3
; CHECK-SAFE-NOT: vrsqrtefp
; CHECK-SAFE: blr
}

