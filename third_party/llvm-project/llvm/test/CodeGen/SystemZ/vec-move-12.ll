; Test insertions of memory values into a nonzero index of an undef.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 insertion into an undef, with an arbitrary index.
define <16 x i8> @f1(i8 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vlrepb %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i8, i8 *%ptr
  %ret = insertelement <16 x i8> undef, i8 %val, i32 12
  ret <16 x i8> %ret
}

; Test v16i8 insertion into an undef, with the first good index for VLVGP.
define <16 x i8> @f2(i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: {{vlrepb|vllezb}} %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i8, i8 *%ptr
  %ret = insertelement <16 x i8> undef, i8 %val, i32 7
  ret <16 x i8> %ret
}

; Test v16i8 insertion into an undef, with the second good index for VLVGP.
define <16 x i8> @f3(i8 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vlrepb %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i8, i8 *%ptr
  %ret = insertelement <16 x i8> undef, i8 %val, i32 15
  ret <16 x i8> %ret
}

; Test v8i16 insertion into an undef, with an arbitrary index.
define <8 x i16> @f4(i16 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: vlreph %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i16, i16 *%ptr
  %ret = insertelement <8 x i16> undef, i16 %val, i32 5
  ret <8 x i16> %ret
}

; Test v8i16 insertion into an undef, with the first good index for VLVGP.
define <8 x i16> @f5(i16 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: {{vlreph|vllezh}} %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i16, i16 *%ptr
  %ret = insertelement <8 x i16> undef, i16 %val, i32 3
  ret <8 x i16> %ret
}

; Test v8i16 insertion into an undef, with the second good index for VLVGP.
define <8 x i16> @f6(i16 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vlreph %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i16, i16 *%ptr
  %ret = insertelement <8 x i16> undef, i16 %val, i32 7
  ret <8 x i16> %ret
}

; Test v4i32 insertion into an undef, with an arbitrary index.
define <4 x i32> @f7(i32 *%ptr) {
; CHECK-LABEL: f7:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i32, i32 *%ptr
  %ret = insertelement <4 x i32> undef, i32 %val, i32 2
  ret <4 x i32> %ret
}

; Test v4i32 insertion into an undef, with the first good index for VLVGP.
define <4 x i32> @f8(i32 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: {{vlrepf|vllezf}} %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i32, i32 *%ptr
  %ret = insertelement <4 x i32> undef, i32 %val, i32 1
  ret <4 x i32> %ret
}

; Test v4i32 insertion into an undef, with the second good index for VLVGP.
define <4 x i32> @f9(i32 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i32, i32 *%ptr
  %ret = insertelement <4 x i32> undef, i32 %val, i32 3
  ret <4 x i32> %ret
}

; Test v2i64 insertion into an undef.
define <2 x i64> @f10(i64 *%ptr) {
; CHECK-LABEL: f10:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i64, i64 *%ptr
  %ret = insertelement <2 x i64> undef, i64 %val, i32 1
  ret <2 x i64> %ret
}

; Test v4f32 insertion into an undef.
define <4 x float> @f11(float *%ptr) {
; CHECK-LABEL: f11:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %val = load float, float *%ptr
  %ret = insertelement <4 x float> undef, float %val, i32 2
  ret <4 x float> %ret
}

; Test v2f64 insertion into an undef.
define <2 x double> @f12(double *%ptr) {
; CHECK-LABEL: f12:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %val = load double, double *%ptr
  %ret = insertelement <2 x double> undef, double %val, i32 1
  ret <2 x double> %ret
}
