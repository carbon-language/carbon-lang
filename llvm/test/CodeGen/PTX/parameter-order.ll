; RUN: llc < %s -march=ptx32 | FileCheck %s

; CHECK: .func (.reg .u32 r0) test_parameter_order (.reg .f32 f1, .reg .u32 r1, .reg .u32 r2, .reg .f32 f2)
define ptx_device i32 @test_parameter_order(float %a, i32 %b, i32 %c, float %d) {
; CHECK: sub.u32 r0, r1, r2
	%result = sub i32 %b, %c
	ret i32 %result
}
