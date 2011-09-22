; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device i16 @t1_u16(i16 %x, i16 %y) {
; CHECK: sub.u16 %ret{{[0-9]+}}, %rh{{[0-9]+}}, %rh{{[0-9]+}};
; CHECK: ret;
	%z = sub i16 %x, %y
	ret i16 %z
}

define ptx_device i32 @t1_u32(i32 %x, i32 %y) {
; CHECK: sub.u32 %ret{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}};
; CHECK: ret;
	%z = sub i32 %x, %y
	ret i32 %z
}

define ptx_device i64 @t1_u64(i64 %x, i64 %y) {
; CHECK: sub.u64 %ret{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}};
; CHECK: ret;
	%z = sub i64 %x, %y
	ret i64 %z
}

define ptx_device float @t1_f32(float %x, float %y) {
; CHECK: sub.rn.f32 %ret{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}
; CHECK: ret;
  %z = fsub float %x, %y
  ret float %z
}

define ptx_device double @t1_f64(double %x, double %y) {
; CHECK: sub.rn.f64 %ret{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}
; CHECK: ret;
  %z = fsub double %x, %y
  ret double %z
}

define ptx_device i16 @t2_u16(i16 %x) {
; CHECK: add.u16 %ret{{[0-9]+}}, %rh{{[0-9]+}}, -1;
; CHECK: ret;
	%z = sub i16 %x, 1
	ret i16 %z
}

define ptx_device i32 @t2_u32(i32 %x) {
; CHECK: add.u32 %ret{{[0-9]+}}, %r{{[0-9]+}}, -1;
; CHECK: ret;
	%z = sub i32 %x, 1
	ret i32 %z
}

define ptx_device i64 @t2_u64(i64 %x) {
; CHECK: add.u64 %ret{{[0-9]+}}, %rd{{[0-9]+}}, -1;
; CHECK: ret;
	%z = sub i64 %x, 1
	ret i64 %z
}

define ptx_device float @t2_f32(float %x) {
; CHECK: add.rn.f32 %ret{{[0-9]+}}, %f{{[0-9]+}}, 0FBF800000;
; CHECK: ret;
  %z = fsub float %x, 1.0
  ret float %z
}

define ptx_device double @t2_f64(double %x) {
; CHECK: add.rn.f64 %ret{{[0-9]+}}, %fd{{[0-9]+}}, 0DBFF0000000000000;
; CHECK: ret;
  %z = fsub double %x, 1.0
  ret double %z
}
