; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device i16 @t1_u16(i16 %x, i16 %y) {
; CHECK: add.u16 rh0, rh1, rh2;
; CHECK-NEXT: ret;
	%z = add i16 %x, %y
	ret i16 %z
}

define ptx_device i32 @t1_u32(i32 %x, i32 %y) {
; CHECK: add.u32 r0, r1, r2;
; CHECK-NEXT: ret;
	%z = add i32 %x, %y
	ret i32 %z
}

define ptx_device i64 @t1_u64(i64 %x, i64 %y) {
; CHECK: add.u64 rd0, rd1, rd2;
; CHECK-NEXT: ret;
	%z = add i64 %x, %y
	ret i64 %z
}

define ptx_device float @t1_f32(float %x, float %y) {
; CHECK: add.rn.f32 r0, r1, r2
; CHECK-NEXT: ret;
  %z = fadd float %x, %y
  ret float %z
}

define ptx_device double @t1_f64(double %x, double %y) {
; CHECK: add.rn.f64 rd0, rd1, rd2
; CHECK-NEXT: ret;
  %z = fadd double %x, %y
  ret double %z
}

define ptx_device i16 @t2_u16(i16 %x) {
; CHECK: add.u16 rh0, rh1, 1;
; CHECK-NEXT: ret;
	%z = add i16 %x, 1
	ret i16 %z
}

define ptx_device i32 @t2_u32(i32 %x) {
; CHECK: add.u32 r0, r1, 1;
; CHECK-NEXT: ret;
	%z = add i32 %x, 1
	ret i32 %z
}

define ptx_device i64 @t2_u64(i64 %x) {
; CHECK: add.u64 rd0, rd1, 1;
; CHECK-NEXT: ret;
	%z = add i64 %x, 1
	ret i64 %z
}

define ptx_device float @t2_f32(float %x) {
; CHECK: add.rn.f32 r0, r1, 0F3F800000;
; CHECK-NEXT: ret;
  %z = fadd float %x, 1.0
  ret float %z
}

define ptx_device double @t2_f64(double %x) {
; CHECK: add.rn.f64 rd0, rd1, 0D3FF0000000000000;
; CHECK-NEXT: ret;
  %z = fadd double %x, 1.0
  ret double %z
}
