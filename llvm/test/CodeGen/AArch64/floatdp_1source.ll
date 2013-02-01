; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@varhalf = global half 0.0
@varfloat = global float 0.0
@vardouble = global double 0.0

declare float @fabsf(float) readonly
declare double @fabs(double) readonly

declare float @llvm.sqrt.f32(float %Val)
declare double @llvm.sqrt.f64(double %Val)

declare float @ceilf(float) readonly
declare double @ceil(double) readonly

declare float @floorf(float) readonly
declare double @floor(double) readonly

declare float @truncf(float) readonly
declare double @trunc(double) readonly

declare float @rintf(float) readonly
declare double @rint(double) readonly

declare float @nearbyintf(float) readonly
declare double @nearbyint(double) readonly

define void @simple_float() {
; CHECK: simple_float:
  %val1 = load volatile float* @varfloat

  %valabs = call float @fabsf(float %val1)
  store volatile float %valabs, float* @varfloat
; CHECK: fabs {{s[0-9]+}}, {{s[0-9]+}}

  %valneg = fsub float -0.0, %val1
  store volatile float %valneg, float* @varfloat
; CHECK: fneg {{s[0-9]+}}, {{s[0-9]+}}

  %valsqrt = call float @llvm.sqrt.f32(float %val1)
  store volatile float %valsqrt, float* @varfloat
; CHECK: fsqrt {{s[0-9]+}}, {{s[0-9]+}}

  %valceil = call float @ceilf(float %val1)
  store volatile float %valceil, float* @varfloat
; CHECK: frintp {{s[0-9]+}}, {{s[0-9]+}}

  %valfloor = call float @floorf(float %val1)
  store volatile float %valfloor, float* @varfloat
; CHECK: frintm {{s[0-9]+}}, {{s[0-9]+}}

  %valtrunc = call float @truncf(float %val1)
  store volatile float %valtrunc, float* @varfloat
; CHECK: frintz {{s[0-9]+}}, {{s[0-9]+}}

  %valrint = call float @rintf(float %val1)
  store volatile float %valrint, float* @varfloat
; CHECK: frintx {{s[0-9]+}}, {{s[0-9]+}}

  %valnearbyint = call float @nearbyintf(float %val1)
  store volatile float %valnearbyint, float* @varfloat
; CHECK: frinti {{s[0-9]+}}, {{s[0-9]+}}

  ret void
}

define void @simple_double() {
; CHECK: simple_double:
  %val1 = load volatile double* @vardouble

  %valabs = call double @fabs(double %val1)
  store volatile double %valabs, double* @vardouble
; CHECK: fabs {{d[0-9]+}}, {{d[0-9]+}}

  %valneg = fsub double -0.0, %val1
  store volatile double %valneg, double* @vardouble
; CHECK: fneg {{d[0-9]+}}, {{d[0-9]+}}

  %valsqrt = call double @llvm.sqrt.f64(double %val1)
  store volatile double %valsqrt, double* @vardouble
; CHECK: fsqrt {{d[0-9]+}}, {{d[0-9]+}}

  %valceil = call double @ceil(double %val1)
  store volatile double %valceil, double* @vardouble
; CHECK: frintp {{d[0-9]+}}, {{d[0-9]+}}

  %valfloor = call double @floor(double %val1)
  store volatile double %valfloor, double* @vardouble
; CHECK: frintm {{d[0-9]+}}, {{d[0-9]+}}

  %valtrunc = call double @trunc(double %val1)
  store volatile double %valtrunc, double* @vardouble
; CHECK: frintz {{d[0-9]+}}, {{d[0-9]+}}

  %valrint = call double @rint(double %val1)
  store volatile double %valrint, double* @vardouble
; CHECK: frintx {{d[0-9]+}}, {{d[0-9]+}}

  %valnearbyint = call double @nearbyint(double %val1)
  store volatile double %valnearbyint, double* @vardouble
; CHECK: frinti {{d[0-9]+}}, {{d[0-9]+}}

  ret void
}

define void @converts() {
; CHECK: converts:

  %val16 = load volatile half* @varhalf
  %val32 = load volatile float* @varfloat
  %val64 = load volatile double* @vardouble

  %val16to32 = fpext half %val16 to float
  store volatile float %val16to32, float* @varfloat
; CHECK: fcvt {{s[0-9]+}}, {{h[0-9]+}}

  %val16to64 = fpext half %val16 to double
  store volatile double %val16to64, double* @vardouble
; CHECK: fcvt {{d[0-9]+}}, {{h[0-9]+}}

  %val32to16 = fptrunc float %val32 to half
  store volatile half %val32to16, half* @varhalf
; CHECK: fcvt {{h[0-9]+}}, {{s[0-9]+}}

  %val32to64 = fpext float %val32 to double
  store volatile double %val32to64, double* @vardouble
; CHECK: fcvt {{d[0-9]+}}, {{s[0-9]+}}

  %val64to16 = fptrunc double %val64 to half
  store volatile half %val64to16, half* @varhalf
; CHECK: fcvt {{h[0-9]+}}, {{d[0-9]+}}

  %val64to32 = fptrunc double %val64 to float
  store volatile float %val64to32, float* @varfloat
; CHECK: fcvt {{s[0-9]+}}, {{d[0-9]+}}

  ret void
}
