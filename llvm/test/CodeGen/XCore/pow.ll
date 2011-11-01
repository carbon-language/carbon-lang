; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.pow.f64(double, double)

define double @test(double %F, double %power) {
; CHECK: test:
; CHECK: bl pow
        %result = call double @llvm.pow.f64(double %F, double %power)
	ret double %result
}

declare float @llvm.pow.f32(float, float)

define float @testf(float %F, float %power) {
; CHECK: testf:
; CHECK: bl powf
        %result = call float @llvm.pow.f32(float %F, float %power)
	ret float %result
}
