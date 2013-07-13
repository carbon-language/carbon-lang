; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@varfloat = global float 0.0
@vardouble = global double 0.0

define void @test_csel(i32 %lhs32, i32 %rhs32, i64 %lhs64) {
; CHECK-LABEL: test_csel:

  %tst1 = icmp ugt i32 %lhs32, %rhs32
  %val1 = select i1 %tst1, float 0.0, float 1.0
  store float %val1, float* @varfloat
; CHECK: ldr [[FLT0:s[0-9]+]], [{{x[0-9]+}}, #:lo12:.LCPI
; CHECK: fmov [[FLT1:s[0-9]+]], #1.0
; CHECK: fcsel {{s[0-9]+}}, [[FLT0]], [[FLT1]], hi

  %rhs64 = sext i32 %rhs32 to i64
  %tst2 = icmp sle i64 %lhs64, %rhs64
  %val2 = select i1 %tst2, double 1.0, double 0.0
  store double %val2, double* @vardouble
; CHECK: ldr [[FLT0:d[0-9]+]], [{{x[0-9]+}}, #:lo12:.LCPI
; CHECK: fmov [[FLT1:d[0-9]+]], #1.0
; CHECK: fcsel {{d[0-9]+}}, [[FLT1]], [[FLT0]], le

  ret void
; CHECK: ret
}
