; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device i16 @t1_u16() {
; CHECK: mov.u16 %ret{{[0-9]+}}, 0;
; CHECK: ret;
	ret i16 0
}

define ptx_device i32 @t1_u32() {
; CHECK: mov.u32 %ret{{[0-9]+}}, 0;
; CHECK: ret;
	ret i32 0
}

define ptx_device i64 @t1_u64() {
; CHECK: mov.u64 %ret{{[0-9]+}}, 0;
; CHECK: ret;
	ret i64 0
}

define ptx_device float @t1_f32() {
; CHECK: mov.f32 %ret{{[0-9]+}}, 0D0000000000000000;
; CHECK: ret;
	ret float 0.0
}

define ptx_device double @t1_f64() {
; CHECK: mov.f64 %ret{{[0-9]+}}, 0D0000000000000000;
; CHECK: ret;
	ret double 0.0
}

define ptx_device i16 @t2_u16(i16 %x) {
; CHECK: mov.b16 %ret{{[0-9]+}}, %param{{[0-9]+}};
; CHECK: ret;
	ret i16 %x
}

define ptx_device i32 @t2_u32(i32 %x) {
; CHECK: mov.b32 %ret{{[0-9]+}}, %param{{[0-9]+}};
; CHECK: ret;
	ret i32 %x
}

define ptx_device i64 @t2_u64(i64 %x) {
; CHECK: mov.b64 %ret{{[0-9]+}}, %param{{[0-9]+}};
; CHECK: ret;
	ret i64 %x
}

define ptx_device float @t3_f32(float %x) {
; CHECK: mov.f32 %ret{{[0-9]+}}, %param{{[0-9]+}};
; CHECK: ret;
	ret float %x
}

define ptx_device double @t3_f64(double %x) {
; CHECK: mov.f64 %ret{{[0-9]+}}, %param{{[0-9]+}};
; CHECK: ret;
	ret double %x
}

