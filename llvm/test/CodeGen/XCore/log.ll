; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.log.f64(double)

define double @test(double %F) {
; CHECK: test:
; CHECK: bl log
        %result = call double @llvm.log.f64(double %F)
	ret double %result
}

declare float @llvm.log.f32(float)

define float @testf(float %F) {
; CHECK: testf:
; CHECK: bl logf
        %result = call float @llvm.log.f32(float %F)
	ret float %result
}
