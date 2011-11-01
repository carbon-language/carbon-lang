; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.exp.f64(double)

define double @test(double %F) {
; CHECK: test:
; CHECK: bl exp
        %result = call double @llvm.exp.f64(double %F)
	ret double %result
}

declare float @llvm.exp.f32(float)

define float @testf(float %F) {
; CHECK: testf:
; CHECK: bl expf
        %result = call float @llvm.exp.f32(float %F)
	ret float %result
}
