; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.cos.f64(double)

define double @test(double %F) {
; CHECK: test:
; CHECK: bl cos
        %result = call double @llvm.cos.f64(double %F)
	ret double %result
}

declare float @llvm.cos.f32(float)

; CHECK: testf:
; CHECK: bl cosf
define float @testf(float %F) {
        %result = call float @llvm.cos.f32(float %F)
	ret float %result
}
