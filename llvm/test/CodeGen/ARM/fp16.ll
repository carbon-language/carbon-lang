; RUN: llc < %s | FileCheck %s
; RUN: llc -mattr=+vfp3,+fp16 < %s | FileCheck --check-prefix=CHECK-FP16 %s
; RUN: llc -mtriple=armv8-eabi < %s | FileCheck --check-prefix=CHECK-ARMV8 %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv7-eabi"

@x = global i16 12902
@y = global i16 0
@z = common global i16 0

define arm_aapcs_vfpcc void @foo() nounwind {
; CHECK-LABEL: foo:
; CHECK-FP16-LABEL: foo:
; CHECK-ARMV8-LABEL: foo:
entry:
  %0 = load i16* @x, align 2
  %1 = load i16* @y, align 2
  %2 = tail call float @llvm.convert.from.fp16.f32(i16 %0)
; CHECK: __gnu_h2f_ieee
; CHECK-FP16: vcvtb.f32.f16
; CHECK-ARMv8: vcvtb.f32.f16
  %3 = tail call float @llvm.convert.from.fp16.f32(i16 %1)
; CHECK: __gnu_h2f_ieee
; CHECK-FP16: vcvtb.f32.f16
; CHECK-ARMV8: vcvtb.f32.f16
  %4 = fadd float %2, %3
  %5 = tail call i16 @llvm.convert.to.fp16.f32(float %4)
; CHECK: __gnu_f2h_ieee
; CHECK-FP16: vcvtb.f16.f32
; CHECK-ARMV8: vcvtb.f16.f32
  store i16 %5, i16* @x, align 2
  ret void
}

define arm_aapcs_vfpcc double @test_from_fp16(i16 %in) {
; CHECK-LABEL: test_from_fp16:
; CHECK-FP-LABEL: test_from_fp16:
; CHECK-ARMV8-LABEL: test_from_fp16:
  %val = call double @llvm.convert.from.fp16.f64(i16 %in)
; CHECK: bl __gnu_h2f_ieee
; CHECK: vmov [[TMP:s[0-9]+]], r0
; CHECK: vcvt.f64.f32 d0, [[TMP]]

; CHECK-FP16: vmov [[TMP16:s[0-9]+]], r0
; CHECK-FP16: vcvtb.f32.f16 [[TMP32:s[0-9]+]], [[TMP16]]
; CHECK-FP16: vcvt.f64.f32 d0, [[TMP32]]

; CHECK-ARMV8: vmov [[TMP:s[0-9]+]], r0
; CHECK-ARMV8: vcvtb.f64.f16 d0, [[TMP]]
  ret double %val
}

define arm_aapcs_vfpcc i16 @test_to_fp16(double %in) {
; CHECK-LABEL: test_to_fp16:
; CHECK-FP-LABEL: test_to_fp16:
; CHECK-ARMV8-LABEL: test_to_fp16:
  %val = call i16 @llvm.convert.to.fp16.f64(double %in)
; CHECK: bl __truncdfhf2

; CHECK-FP16: bl __truncdfhf2

; CHECK-ARMV8: vcvtb.f16.f64 [[TMP:s[0-9]+]], d0
; CHECK-ARMV8: vmov r0, [[TMP]]
  ret i16 %val
}

declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone
declare double @llvm.convert.from.fp16.f64(i16) nounwind readnone

declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone
declare i16 @llvm.convert.to.fp16.f64(double) nounwind readnone
