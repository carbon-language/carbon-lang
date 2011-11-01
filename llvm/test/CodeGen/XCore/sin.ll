; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.sin.f64(double)

define double @test(double %F) {
; CHECK: test:
; CHECK: bl sin
        %result = call double @llvm.sin.f64(double %F)
	ret double %result
}

declare float @llvm.sin.f32(float)

define float @testf(float %F) {
; CHECK: testf:
; CHECK: bl sinf
        %result = call float @llvm.sin.f32(float %F)
	ret float %result
}
