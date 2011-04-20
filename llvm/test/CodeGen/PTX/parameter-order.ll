; RUN: llc < %s -march=ptx32 | FileCheck %s

; CHECK: .func (.reg .u32 r0) test_parameter_order (.reg .u32 r1, .reg .u32 r2)
define ptx_device i32 @test_parameter_order(i32 %x, i32 %y) {
; CHECK: sub.u32 r0, r1, r2
	%z = sub i32 %x, %y
	ret i32 %z
}
