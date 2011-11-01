; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.sqrt.f64(double)

define double @test(double %F) {
; CHECK: test:
; CHECK: bl sqrt
        %result = call double @llvm.sqrt.f64(double %F)
	ret double %result
}

declare float @llvm.sqrt.f32(float)

define float @testf(float %F) {
; CHECK: testf:
; CHECK: bl sqrtf
        %result = call float @llvm.sqrt.f32(float %F)
	ret float %result
}
