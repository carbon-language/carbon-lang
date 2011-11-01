; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.exp2.f64(double)

define double @test(double %F) {
; CHECK: test:
; CHECK: bl exp2
        %result = call double @llvm.exp2.f64(double %F)
	ret double %result
}

declare float @llvm.exp2.f32(float)

define float @testf(float %F) {
; CHECK: testf:
; CHECK: bl exp2f
        %result = call float @llvm.exp2.f32(float %F)
	ret float %result
}
