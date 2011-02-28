; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_device i32 @t1(i32 %x, i32 %y) {
;CHECK: sub.s32 r0, r1, r2;
	%z = sub i32 %x, %y
;CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t2(i32 %x) {
;CHECK: add.s32 r0, r1, -1;
	%z = sub i32 %x, 1
;CHECK: ret;
	ret i32 %z
}

define ptx_device float @t3(float %x, float %y) {
; CHECK: sub.f32 f0, f1, f2
; CHECK-NEXT: ret;
  %z = fsub float %x, %y
  ret float %z
}

define ptx_device float @t4(float %x) {
; CHECK: add.f32 f0, f1, 0FBF800000;
; CHECK-NEXT: ret;
  %z = fsub float %x, 1.0
  ret float %z
}
