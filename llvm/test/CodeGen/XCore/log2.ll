; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.log2.f64(double)

define double @test(double %F) {
; CHECK: test:
; CHECK: bl log2
        %result = call double @llvm.log2.f64(double %F)
	ret double %result
}

declare float @llvm.log2.f32(float)

define float @testf(float %F) {
; CHECK: testf:
; CHECK: bl log2f
        %result = call float @llvm.log2.f32(float %F)
	ret float %result
}
