; RUN: llc < %s -march=ptx32 | FileCheck %s

; preds 
; (note: we convert back to i32 to return)

define ptx_device i32 @cvt_pred_i16(i16 %x, i1 %y) {
; CHECK: setp.gt.b16 p0, rh1, 0
; CHECK-NEXT: and.pred p0, p0, p1;
; CHECK-NEXT: selp.u32 r0, 1, 0, p0;
; CHECK-NEXT: ret;
	%a = trunc i16 %x to i1
	%b = and i1 %a, %y
	%c = zext i1 %b to i32
	ret i32 %c
}

define ptx_device i32 @cvt_pred_i32(i32 %x, i1 %y) {
; CHECK: setp.gt.b32 p0, r1, 0
; CHECK-NEXT: and.pred p0, p0, p1;
; CHECK-NEXT: selp.u32 r0, 1, 0, p0;
; CHECK-NEXT: ret;
	%a = trunc i32 %x to i1
	%b = and i1 %a, %y
	%c = zext i1 %b to i32
	ret i32 %c
}

define ptx_device i32 @cvt_pred_i64(i64 %x, i1 %y) {
; CHECK: setp.gt.b64 p0, rd1, 0
; CHECK-NEXT: and.pred p0, p0, p1;
; CHECK-NEXT: selp.u32 r0, 1, 0, p0;
; CHECK-NEXT: ret;
	%a = trunc i64 %x to i1
	%b = and i1 %a, %y
	%c = zext i1 %b to i32
	ret i32 %c
}

define ptx_device i32 @cvt_pred_f32(float %x, i1 %y) {
; CHECK: setp.gt.b32 p0, r1, 0
; CHECK-NEXT: and.pred p0, p0, p1;
; CHECK-NEXT: selp.u32 r0, 1, 0, p0;
; CHECK-NEXT: ret;
	%a = fptoui float %x to i1
	%b = and i1 %a, %y
	%c = zext i1 %b to i32
	ret i32 %c
}

define ptx_device i32 @cvt_pred_f64(double %x, i1 %y) {
; CHECK: setp.gt.b64 p0, rd1, 0
; CHECK-NEXT: and.pred p0, p0, p1;
; CHECK-NEXT: selp.u32 r0, 1, 0, p0;
; CHECK-NEXT: ret;
	%a = fptoui double %x to i1
	%b = and i1 %a, %y
	%c = zext i1 %b to i32
	ret i32 %c
}

; i16

define ptx_device i16 @cvt_i16_preds(i1 %x) {
; CHECK: selp.u16 rh0, 1, 0, p1;
; CHECK-NEXT: ret;
	%a = zext i1 %x to i16
	ret i16 %a
}

define ptx_device i16 @cvt_i16_i32(i32 %x) {
; CHECK: cvt.u16.u32 rh0, r1;
; CHECK-NEXT: ret;
	%a = trunc i32 %x to i16
	ret i16 %a
}

define ptx_device i16 @cvt_i16_i64(i64 %x) {
; CHECK: cvt.u16.u64 rh0, rd1;
; CHECK-NEXT: ret;
	%a = trunc i64 %x to i16
	ret i16 %a
}

define ptx_device i16 @cvt_i16_f32(float %x) {
; CHECK: cvt.rzi.u16.f32 rh0, r1;
; CHECK-NEXT: ret;
	%a = fptoui float %x to i16
	ret i16 %a
}

define ptx_device i16 @cvt_i16_f64(double %x) {
; CHECK: cvt.rzi.u16.f64 rh0, rd1;
; CHECK-NEXT: ret;
	%a = fptoui double %x to i16
	ret i16 %a
}

; i32

define ptx_device i32 @cvt_i32_preds(i1 %x) {
; CHECK: selp.u32 r0, 1, 0, p1;
; CHECK-NEXT: ret;
	%a = zext i1 %x to i32
	ret i32 %a
}

define ptx_device i32 @cvt_i32_i16(i16 %x) {
; CHECK: cvt.u32.u16 r0, rh1;
; CHECK-NEXT: ret;
	%a = zext i16 %x to i32
	ret i32 %a
}

define ptx_device i32 @cvt_i32_i64(i64 %x) {
; CHECK: cvt.u32.u64 r0, rd1;
; CHECK-NEXT: ret;
	%a = trunc i64 %x to i32
	ret i32 %a
}

define ptx_device i32 @cvt_i32_f32(float %x) {
; CHECK: cvt.rzi.u32.f32 r0, r1;
; CHECK-NEXT: ret;
	%a = fptoui float %x to i32
	ret i32 %a
}

define ptx_device i32 @cvt_i32_f64(double %x) {
; CHECK: cvt.rzi.u32.f64 r0, rd1;
; CHECK-NEXT: ret;
	%a = fptoui double %x to i32
	ret i32 %a
}

; i64

define ptx_device i64 @cvt_i64_preds(i1 %x) {
; CHECK: selp.u64 rd0, 1, 0, p1;
; CHECK-NEXT: ret;
	%a = zext i1 %x to i64
	ret i64 %a
}

define ptx_device i64 @cvt_i64_i16(i16 %x) {
; CHECK: cvt.u64.u16 rd0, rh1;
; CHECK-NEXT: ret;
	%a = zext i16 %x to i64
	ret i64 %a
}

define ptx_device i64 @cvt_i64_i32(i32 %x) {
; CHECK: cvt.u64.u32 rd0, r1;
; CHECK-NEXT: ret;
	%a = zext i32 %x to i64
	ret i64 %a
}

define ptx_device i64 @cvt_i64_f32(float %x) {
; CHECK: cvt.rzi.u64.f32 rd0, r1;
; CHECK-NEXT: ret;
	%a = fptoui float %x to i64
	ret i64 %a
}

define ptx_device i64 @cvt_i64_f64(double %x) {
; CHECK: cvt.rzi.u64.f64 rd0, rd1;
; CHECK: ret;
	%a = fptoui double %x to i64
	ret i64 %a
}

; f32

define ptx_device float @cvt_f32_preds(i1 %x) {
; CHECK: selp.f32 r0, 0F3F800000, 0F00000000, p1;
; CHECK-NEXT: ret;
	%a = uitofp i1 %x to float
	ret float %a
}

define ptx_device float @cvt_f32_i16(i16 %x) {
; CHECK: cvt.rn.f32.u16 r0, rh1;
; CHECK-NEXT: ret;
	%a = uitofp i16 %x to float
	ret float %a
}

define ptx_device float @cvt_f32_i32(i32 %x) {
; CHECK: cvt.rn.f32.u32 r0, r1;
; CHECK-NEXT: ret;
	%a = uitofp i32 %x to float
	ret float %a
}

define ptx_device float @cvt_f32_i64(i64 %x) {
; CHECK: cvt.rn.f32.u64 r0, rd1;
; CHECK-NEXT: ret;
	%a = uitofp i64 %x to float
	ret float %a
}

define ptx_device float @cvt_f32_f64(double %x) {
; CHECK: cvt.rn.f32.f64 r0, rd1;
; CHECK-NEXT: ret;
	%a = fptrunc double %x to float
	ret float %a
}

; f64

define ptx_device double @cvt_f64_preds(i1 %x) {
; CHECK: selp.f64 rd0, 0D3F80000000000000, 0D0000000000000000, p1;
; CHECK-NEXT: ret;
	%a = uitofp i1 %x to double
	ret double %a
}

define ptx_device double @cvt_f64_i16(i16 %x) {
; CHECK: cvt.rn.f64.u16 rd0, rh1;
; CHECK-NEXT: ret;
	%a = uitofp i16 %x to double
	ret double %a
}

define ptx_device double @cvt_f64_i32(i32 %x) {
; CHECK: cvt.rn.f64.u32 rd0, r1;
; CHECK-NEXT: ret;
	%a = uitofp i32 %x to double
	ret double %a
}

define ptx_device double @cvt_f64_i64(i64 %x) {
; CHECK: cvt.rn.f64.u64 rd0, rd1;
; CHECK-NEXT: ret;
	%a = uitofp i64 %x to double
	ret double %a
}

define ptx_device double @cvt_f64_f32(float %x) {
; CHECK: cvt.f64.f32 rd0, r1;
; CHECK-NEXT: ret;
	%a = fpext float %x to double
	ret double %a
}
