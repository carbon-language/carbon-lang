; RUN: llc < %s -march=ptx32 -mattr=+sm10 | FileCheck %s

define ptx_device float @t1_f32(float %x, float %y) {
; CHECK: div.f32 %ret{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: ret;
	%a = fdiv float %x, %y
	ret float %a
}

define ptx_device double @t1_f64(double %x, double %y) {
; CHECK: div.f64 %ret{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK: ret;
	%a = fdiv double %x, %y
	ret double %a
}
