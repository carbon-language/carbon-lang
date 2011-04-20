; RUN: llc < %s -march=ptx32 -mattr=+sm10 | FileCheck %s

define ptx_device float @t1_f32(float %x, float %y) {
; CHECK: div.approx.f32 f0, f1, f2;
; CHECK-NEXT: ret;
	%a = fdiv float %x, %y
	ret float %a
}

define ptx_device double @t1_f64(double %x, double %y) {
; CHECK: div.f64 fd0, fd1, fd2;
; CHECK-NEXT: ret;
	%a = fdiv double %x, %y
	ret double %a
}
