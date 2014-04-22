; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; arm64 does not use intrinsics for comparisons.

;; Scalar Floating-point Compare

define i32 @test_vceqs_f32(float %a, float %b) {
; CHECK-LABEL: test_vceqs_f32
; CHECK: fcmeq {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fceq2.i = call <1 x i32> @llvm.aarch64.neon.fceq.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fceq2.i, i32 0
  ret i32 %0
}

define i64 @test_vceqd_f64(double %a, double %b) {
; CHECK-LABEL: test_vceqd_f64
; CHECK: fcmeq {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fceq2.i = call <1 x i64> @llvm.aarch64.neon.fceq.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fceq2.i, i32 0
  ret i64 %0
}

define <1 x i64> @test_vceqz_f64(<1 x double> %a) {
; CHECK-LABEL: test_vceqz_f64
; CHECK: fcmeq  {{d[0-9]+}}, {{d[0-9]+}}, #0.0
entry:
  %0 = fcmp oeq <1 x double> %a, zeroinitializer
  %vceqz.i = sext <1 x i1> %0 to <1 x i64>
  ret <1 x i64> %vceqz.i
}

define i32 @test_vceqzs_f32(float %a) {
; CHECK-LABEL: test_vceqzs_f32
; CHECK: fcmeq {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %fceq1.i = call <1 x i32> @llvm.aarch64.neon.fceq.v1i32.f32.f32(float %a, float 0.0)
  %0 = extractelement <1 x i32> %fceq1.i, i32 0
  ret i32 %0
}

define i64 @test_vceqzd_f64(double %a) {
; CHECK-LABEL: test_vceqzd_f64
; CHECK: fcmeq {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %fceq1.i = call <1 x i64> @llvm.aarch64.neon.fceq.v1i64.f64.f32(double %a, float 0.0)
  %0 = extractelement <1 x i64> %fceq1.i, i32 0
  ret i64 %0
}

define i32 @test_vcges_f32(float %a, float %b) {
; CHECK-LABEL: test_vcges_f32
; CHECK: fcmge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcge2.i = call <1 x i32> @llvm.aarch64.neon.fcge.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcge2.i, i32 0
  ret i32 %0
}

define i64 @test_vcged_f64(double %a, double %b) {
; CHECK-LABEL: test_vcged_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcge2.i = call <1 x i64> @llvm.aarch64.neon.fcge.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcge2.i, i32 0
  ret i64 %0
}

define i32 @test_vcgezs_f32(float %a) {
; CHECK-LABEL: test_vcgezs_f32
; CHECK: fcmge {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %fcge1.i = call <1 x i32> @llvm.aarch64.neon.fcge.v1i32.f32.f32(float %a, float 0.0)
  %0 = extractelement <1 x i32> %fcge1.i, i32 0
  ret i32 %0
}

define i64 @test_vcgezd_f64(double %a) {
; CHECK-LABEL: test_vcgezd_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %fcge1.i = call <1 x i64> @llvm.aarch64.neon.fcge.v1i64.f64.f32(double %a, float 0.0)
  %0 = extractelement <1 x i64> %fcge1.i, i32 0
  ret i64 %0
}

define i32 @test_vcgts_f32(float %a, float %b) {
; CHECK-LABEL: test_vcgts_f32
; CHECK: fcmgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcgt2.i = call <1 x i32> @llvm.aarch64.neon.fcgt.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcgt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcgtd_f64(double %a, double %b) {
; CHECK-LABEL: test_vcgtd_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcgt2.i = call <1 x i64> @llvm.aarch64.neon.fcgt.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcgt2.i, i32 0
  ret i64 %0
}

define i32 @test_vcgtzs_f32(float %a) {
; CHECK-LABEL: test_vcgtzs_f32
; CHECK: fcmgt {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %fcgt1.i = call <1 x i32> @llvm.aarch64.neon.fcgt.v1i32.f32.f32(float %a, float 0.0)
  %0 = extractelement <1 x i32> %fcgt1.i, i32 0
  ret i32 %0
}

define i64 @test_vcgtzd_f64(double %a) {
; CHECK-LABEL: test_vcgtzd_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %fcgt1.i = call <1 x i64> @llvm.aarch64.neon.fcgt.v1i64.f64.f32(double %a, float 0.0)
  %0 = extractelement <1 x i64> %fcgt1.i, i32 0
  ret i64 %0
}

define i32 @test_vcles_f32(float %a, float %b) {
; CHECK-LABEL: test_vcles_f32
; CHECK: fcmge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcge2.i = call <1 x i32> @llvm.aarch64.neon.fcge.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcge2.i, i32 0
  ret i32 %0
}

define i64 @test_vcled_f64(double %a, double %b) {
; CHECK-LABEL: test_vcled_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcge2.i = call <1 x i64> @llvm.aarch64.neon.fcge.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcge2.i, i32 0
  ret i64 %0
}

define i32 @test_vclezs_f32(float %a) {
; CHECK-LABEL: test_vclezs_f32
; CHECK: fcmle {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %fcle1.i = call <1 x i32> @llvm.aarch64.neon.fclez.v1i32.f32.f32(float %a, float 0.0)
  %0 = extractelement <1 x i32> %fcle1.i, i32 0
  ret i32 %0
}

define i64 @test_vclezd_f64(double %a) {
; CHECK-LABEL: test_vclezd_f64
; CHECK: fcmle {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %fcle1.i = call <1 x i64> @llvm.aarch64.neon.fclez.v1i64.f64.f32(double %a, float 0.0)
  %0 = extractelement <1 x i64> %fcle1.i, i32 0
  ret i64 %0
}

define i32 @test_vclts_f32(float %a, float %b) {
; CHECK-LABEL: test_vclts_f32
; CHECK: fcmgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcgt2.i = call <1 x i32> @llvm.aarch64.neon.fcgt.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcgt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcltd_f64(double %a, double %b) {
; CHECK-LABEL: test_vcltd_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcgt2.i = call <1 x i64> @llvm.aarch64.neon.fcgt.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcgt2.i, i32 0
  ret i64 %0
}

define i32 @test_vcltzs_f32(float %a) {
; CHECK-LABEL: test_vcltzs_f32
; CHECK: fcmlt {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %fclt1.i = call <1 x i32> @llvm.aarch64.neon.fcltz.v1i32.f32.f32(float %a, float 0.0)
  %0 = extractelement <1 x i32> %fclt1.i, i32 0
  ret i32 %0
}

define i64 @test_vcltzd_f64(double %a) {
; CHECK-LABEL: test_vcltzd_f64
; CHECK: fcmlt {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %fclt1.i = call <1 x i64> @llvm.aarch64.neon.fcltz.v1i64.f64.f32(double %a, float 0.0)
  %0 = extractelement <1 x i64> %fclt1.i, i32 0
  ret i64 %0
}

define i32 @test_vcages_f32(float %a, float %b) {
; CHECK-LABEL: test_vcages_f32
; CHECK: facge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcage2.i = call <1 x i32> @llvm.aarch64.neon.fcage.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcage2.i, i32 0
  ret i32 %0
}

define i64 @test_vcaged_f64(double %a, double %b) {
; CHECK-LABEL: test_vcaged_f64
; CHECK: facge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcage2.i = call <1 x i64> @llvm.aarch64.neon.fcage.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcage2.i, i32 0
  ret i64 %0
}

define i32 @test_vcagts_f32(float %a, float %b) {
; CHECK-LABEL: test_vcagts_f32
; CHECK: facgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcagt2.i = call <1 x i32> @llvm.aarch64.neon.fcagt.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcagt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcagtd_f64(double %a, double %b) {
; CHECK-LABEL: test_vcagtd_f64
; CHECK: facgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcagt2.i = call <1 x i64> @llvm.aarch64.neon.fcagt.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcagt2.i, i32 0
  ret i64 %0
}

define i32 @test_vcales_f32(float %a, float %b) {
; CHECK-LABEL: test_vcales_f32
; CHECK: facge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcage2.i = call <1 x i32> @llvm.aarch64.neon.fcage.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcage2.i, i32 0
  ret i32 %0
}

define i64 @test_vcaled_f64(double %a, double %b) {
; CHECK-LABEL: test_vcaled_f64
; CHECK: facge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcage2.i = call <1 x i64> @llvm.aarch64.neon.fcage.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcage2.i, i32 0
  ret i64 %0
}

define i32 @test_vcalts_f32(float %a, float %b) {
; CHECK-LABEL: test_vcalts_f32
; CHECK: facgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %fcalt2.i = call <1 x i32> @llvm.aarch64.neon.fcagt.v1i32.f32.f32(float %a, float %b)
  %0 = extractelement <1 x i32> %fcalt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcaltd_f64(double %a, double %b) {
; CHECK-LABEL: test_vcaltd_f64
; CHECK: facgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %fcalt2.i = call <1 x i64> @llvm.aarch64.neon.fcagt.v1i64.f64.f64(double %a, double %b)
  %0 = extractelement <1 x i64> %fcalt2.i, i32 0
  ret i64 %0
}

declare <1 x i32> @llvm.aarch64.neon.fceq.v1i32.f32.f32(float, float)
declare <1 x i64> @llvm.aarch64.neon.fceq.v1i64.f64.f32(double, float)
declare <1 x i64> @llvm.aarch64.neon.fceq.v1i64.f64.f64(double, double)
declare <1 x i32> @llvm.aarch64.neon.fcge.v1i32.f32.f32(float, float)
declare <1 x i64> @llvm.aarch64.neon.fcge.v1i64.f64.f32(double, float)
declare <1 x i64> @llvm.aarch64.neon.fcge.v1i64.f64.f64(double, double)
declare <1 x i32> @llvm.aarch64.neon.fclez.v1i32.f32.f32(float, float)
declare <1 x i64> @llvm.aarch64.neon.fclez.v1i64.f64.f32(double, float)
declare <1 x i32> @llvm.aarch64.neon.fcgt.v1i32.f32.f32(float, float)
declare <1 x i64> @llvm.aarch64.neon.fcgt.v1i64.f64.f32(double, float)
declare <1 x i64> @llvm.aarch64.neon.fcgt.v1i64.f64.f64(double, double)
declare <1 x i32> @llvm.aarch64.neon.fcltz.v1i32.f32.f32(float, float)
declare <1 x i64> @llvm.aarch64.neon.fcltz.v1i64.f64.f32(double, float)
declare <1 x i32> @llvm.aarch64.neon.fcage.v1i32.f32.f32(float, float)
declare <1 x i64> @llvm.aarch64.neon.fcage.v1i64.f64.f64(double, double)
declare <1 x i32> @llvm.aarch64.neon.fcagt.v1i32.f32.f32(float, float)
declare <1 x i64> @llvm.aarch64.neon.fcagt.v1i64.f64.f64(double, double)
