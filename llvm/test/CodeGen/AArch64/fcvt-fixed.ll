; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -O0 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=arm64-apple-ios7.0 -O0

; (The O0 test is to make sure FastISel still constrains its operands properly
; and the verifier doesn't trigger).

@var32 = global i32 0
@var64 = global i64 0

define void @test_fcvtzs(float %flt, double %dbl) {
; CHECK-LABEL: test_fcvtzs:

  %fix1 = fmul float %flt, 128.0
  %cvt1 = fptosi float %fix1 to i32
; CHECK: fcvtzs {{w[0-9]+}}, {{s[0-9]+}}, #7
  store volatile i32 %cvt1, i32* @var32

  %fix2 = fmul float %flt, 4294967296.0
  %cvt2 = fptosi float %fix2 to i32
; CHECK: fcvtzs {{w[0-9]+}}, {{s[0-9]+}}, #32
  store volatile i32 %cvt2, i32* @var32

  %fix3 = fmul float %flt, 128.0
  %cvt3 = fptosi float %fix3 to i64
; CHECK: fcvtzs {{x[0-9]+}}, {{s[0-9]+}}, #7
  store volatile i64 %cvt3, i64* @var64

  %fix4 = fmul float %flt, 18446744073709551616.0
  %cvt4 = fptosi float %fix4 to i64
; CHECK: fcvtzs {{x[0-9]+}}, {{s[0-9]+}}, #64
  store volatile i64 %cvt4, i64* @var64

  %fix5 = fmul double %dbl, 128.0
  %cvt5 = fptosi double %fix5 to i32
; CHECK: fcvtzs {{w[0-9]+}}, {{d[0-9]+}}, #7
  store volatile i32 %cvt5, i32* @var32

  %fix6 = fmul double %dbl, 4294967296.0
  %cvt6 = fptosi double %fix6 to i32
; CHECK: fcvtzs {{w[0-9]+}}, {{d[0-9]+}}, #32
  store volatile i32 %cvt6, i32* @var32

  %fix7 = fmul double %dbl, 128.0
  %cvt7 = fptosi double %fix7 to i64
; CHECK: fcvtzs {{x[0-9]+}}, {{d[0-9]+}}, #7
  store volatile i64 %cvt7, i64* @var64

  %fix8 = fmul double %dbl, 18446744073709551616.0
  %cvt8 = fptosi double %fix8 to i64
; CHECK: fcvtzs {{x[0-9]+}}, {{d[0-9]+}}, #64
  store volatile i64 %cvt8, i64* @var64

  ret void
}

define void @test_fcvtzu(float %flt, double %dbl) {
; CHECK-LABEL: test_fcvtzu:

  %fix1 = fmul float %flt, 128.0
  %cvt1 = fptoui float %fix1 to i32
; CHECK: fcvtzu {{w[0-9]+}}, {{s[0-9]+}}, #7
  store volatile i32 %cvt1, i32* @var32

  %fix2 = fmul float %flt, 4294967296.0
  %cvt2 = fptoui float %fix2 to i32
; CHECK: fcvtzu {{w[0-9]+}}, {{s[0-9]+}}, #32
  store volatile i32 %cvt2, i32* @var32

  %fix3 = fmul float %flt, 128.0
  %cvt3 = fptoui float %fix3 to i64
; CHECK: fcvtzu {{x[0-9]+}}, {{s[0-9]+}}, #7
  store volatile i64 %cvt3, i64* @var64

  %fix4 = fmul float %flt, 18446744073709551616.0
  %cvt4 = fptoui float %fix4 to i64
; CHECK: fcvtzu {{x[0-9]+}}, {{s[0-9]+}}, #64
  store volatile i64 %cvt4, i64* @var64

  %fix5 = fmul double %dbl, 128.0
  %cvt5 = fptoui double %fix5 to i32
; CHECK: fcvtzu {{w[0-9]+}}, {{d[0-9]+}}, #7
  store volatile i32 %cvt5, i32* @var32

  %fix6 = fmul double %dbl, 4294967296.0
  %cvt6 = fptoui double %fix6 to i32
; CHECK: fcvtzu {{w[0-9]+}}, {{d[0-9]+}}, #32
  store volatile i32 %cvt6, i32* @var32

  %fix7 = fmul double %dbl, 128.0
  %cvt7 = fptoui double %fix7 to i64
; CHECK: fcvtzu {{x[0-9]+}}, {{d[0-9]+}}, #7
  store volatile i64 %cvt7, i64* @var64

  %fix8 = fmul double %dbl, 18446744073709551616.0
  %cvt8 = fptoui double %fix8 to i64
; CHECK: fcvtzu {{x[0-9]+}}, {{d[0-9]+}}, #64
  store volatile i64 %cvt8, i64* @var64

  ret void
}

@varfloat = global float 0.0
@vardouble = global double 0.0

define void @test_scvtf(i32 %int, i64 %long) {
; CHECK-LABEL: test_scvtf:

  %cvt1 = sitofp i32 %int to float
  %fix1 = fdiv float %cvt1, 128.0
; CHECK: scvtf {{s[0-9]+}}, {{w[0-9]+}}, #7
  store volatile float %fix1, float* @varfloat

  %cvt2 = sitofp i32 %int to float
  %fix2 = fdiv float %cvt2, 4294967296.0
; CHECK: scvtf {{s[0-9]+}}, {{w[0-9]+}}, #32
  store volatile float %fix2, float* @varfloat

  %cvt3 = sitofp i64 %long to float
  %fix3 = fdiv float %cvt3, 128.0
; CHECK: scvtf {{s[0-9]+}}, {{x[0-9]+}}, #7
  store volatile float %fix3, float* @varfloat

  %cvt4 = sitofp i64 %long to float
  %fix4 = fdiv float %cvt4, 18446744073709551616.0
; CHECK: scvtf {{s[0-9]+}}, {{x[0-9]+}}, #64
  store volatile float %fix4, float* @varfloat

  %cvt5 = sitofp i32 %int to double
  %fix5 = fdiv double %cvt5, 128.0
; CHECK: scvtf {{d[0-9]+}}, {{w[0-9]+}}, #7
  store volatile double %fix5, double* @vardouble

  %cvt6 = sitofp i32 %int to double
  %fix6 = fdiv double %cvt6, 4294967296.0
; CHECK: scvtf {{d[0-9]+}}, {{w[0-9]+}}, #32
  store volatile double %fix6, double* @vardouble

  %cvt7 = sitofp i64 %long to double
  %fix7 = fdiv double %cvt7, 128.0
; CHECK: scvtf {{d[0-9]+}}, {{x[0-9]+}}, #7
  store volatile double %fix7, double* @vardouble

  %cvt8 = sitofp i64 %long to double
  %fix8 = fdiv double %cvt8, 18446744073709551616.0
; CHECK: scvtf {{d[0-9]+}}, {{x[0-9]+}}, #64
  store volatile double %fix8, double* @vardouble

  ret void
}

define void @test_ucvtf(i32 %int, i64 %long) {
; CHECK-LABEL: test_ucvtf:

  %cvt1 = uitofp i32 %int to float
  %fix1 = fdiv float %cvt1, 128.0
; CHECK: ucvtf {{s[0-9]+}}, {{w[0-9]+}}, #7
  store volatile float %fix1, float* @varfloat

  %cvt2 = uitofp i32 %int to float
  %fix2 = fdiv float %cvt2, 4294967296.0
; CHECK: ucvtf {{s[0-9]+}}, {{w[0-9]+}}, #32
  store volatile float %fix2, float* @varfloat

  %cvt3 = uitofp i64 %long to float
  %fix3 = fdiv float %cvt3, 128.0
; CHECK: ucvtf {{s[0-9]+}}, {{x[0-9]+}}, #7
  store volatile float %fix3, float* @varfloat

  %cvt4 = uitofp i64 %long to float
  %fix4 = fdiv float %cvt4, 18446744073709551616.0
; CHECK: ucvtf {{s[0-9]+}}, {{x[0-9]+}}, #64
  store volatile float %fix4, float* @varfloat

  %cvt5 = uitofp i32 %int to double
  %fix5 = fdiv double %cvt5, 128.0
; CHECK: ucvtf {{d[0-9]+}}, {{w[0-9]+}}, #7
  store volatile double %fix5, double* @vardouble

  %cvt6 = uitofp i32 %int to double
  %fix6 = fdiv double %cvt6, 4294967296.0
; CHECK: ucvtf {{d[0-9]+}}, {{w[0-9]+}}, #32
  store volatile double %fix6, double* @vardouble

  %cvt7 = uitofp i64 %long to double
  %fix7 = fdiv double %cvt7, 128.0
; CHECK: ucvtf {{d[0-9]+}}, {{x[0-9]+}}, #7
  store volatile double %fix7, double* @vardouble

  %cvt8 = uitofp i64 %long to double
  %fix8 = fdiv double %cvt8, 18446744073709551616.0
; CHECK: ucvtf {{d[0-9]+}}, {{x[0-9]+}}, #64
  store volatile double %fix8, double* @vardouble

  ret void
}
