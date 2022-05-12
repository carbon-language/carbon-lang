; RUN: opt < %s -inline -instcombine -S | FileCheck %s

; PR22857: http://llvm.org/bugs/show_bug.cgi?id=22857
; The inliner should not add an edge to an intrinsic and
; then assert that it did not add an edge to an intrinsic!

define float @foo(float %f1) {
  %call = call float @bar(float %f1)
  ret float %call

; CHECK-LABEL: @foo(
; CHECK-NEXT: call fast float @llvm.fabs.f32
; CHECK-NEXT: ret float
}

define float @bar(float %f1) {
  %call = call float @sqr(float %f1)
  %call1 = call fast float @sqrtf(float %call)
  ret float %call1
}

define float @sqr(float %f) {
  %mul = fmul fast float %f, %f
  ret float %mul
}

declare float @sqrtf(float)

