; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.powi.f64(double, i32)

define double @test(double %F, i32 %power) {
; CHECK: test:
; CHECK: bl __powidf2
        %result = call double @llvm.powi.f64(double %F, i32 %power)
	ret double %result
}

declare float @llvm.powi.f32(float, i32)

define float @testf(float %F, i32 %power) {
; CHECK: testf:
; CHECK: bl __powisf2
        %result = call float @llvm.powi.f32(float %F, i32 %power)
	ret float %result
}
