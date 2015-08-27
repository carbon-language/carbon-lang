; RUN: opt < %s -constprop -S | FileCheck %s
; RUN: opt < %s -constprop -disable-simplify-libcalls -S | FileCheck %s --check-prefix=FNOBUILTIN

declare double @acos(double)
declare double @asin(double)
declare double @atan(double)
declare double @atan2(double, double)
declare double @ceil(double)
declare double @cos(double)
declare double @cosh(double)
declare double @exp(double)
declare double @exp2(double)
declare double @fabs(double)
declare double @floor(double)
declare double @fmod(double, double)
declare double @log(double)
declare double @log10(double)
declare double @pow(double, double)
declare double @sin(double)
declare double @sinh(double)
declare double @sqrt(double)
declare double @tan(double)
declare double @tanh(double)

declare float @acosf(float)
declare float @asinf(float)
declare float @atanf(float)
declare float @atan2f(float, float)
declare float @ceilf(float)
declare float @cosf(float)
declare float @coshf(float)
declare float @expf(float)
declare float @exp2f(float)
declare float @fabsf(float)
declare float @floorf(float)
declare float @fmodf(float, float)
declare float @logf(float)
declare float @log10f(float)
declare float @powf(float, float)
declare float @sinf(float)
declare float @sinhf(float)
declare float @sqrtf(float)
declare float @tanf(float)
declare float @tanhf(float)

define double @T() {
; CHECK-LABEL: @T(
; CHECK-NOT: call
; CHECK: ret
  %A = call double @cos(double 0.000000e+00)
  %B = call double @sin(double 0.000000e+00)
  %a = fadd double %A, %B
  %C = call double @tan(double 0.000000e+00)
  %b = fadd double %a, %C
  %D = call double @sqrt(double 4.000000e+00)
  %c = fadd double %b, %D

  ; PR9315
  %E = call double @exp2(double 4.0)
  %d = fadd double %c, %E 
  ret double %d
}

define i1 @test_sse_cvt() nounwind readnone {
; CHECK-LABEL: @test_sse_cvt(
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i1 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i2 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i3 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i4 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> <double 1.75, double undef>) nounwind
  %i5 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> <double 1.75, double undef>) nounwind
  %i6 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> <double 1.75, double undef>) nounwind
  %i7 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> <double 1.75, double undef>) nounwind
  %sum11 = add i32 %i0, %i1
  %sum12 = add i32 %i4, %i5
  %sum1 = add i32 %sum11, %sum12
  %sum21 = add i64 %i2, %i3
  %sum22 = add i64 %i6, %i7
  %sum2 = add i64 %sum21, %sum22
  %sum1.sext = sext i32 %sum1 to i64
  %b = icmp eq i64 %sum1.sext, %sum2
  ret i1 %b
}

declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone

define double @test_intrinsic_pow() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_intrinsic_pow(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @llvm.pow.f64(double 1.500000e+00, double 3.000000e+00)
  ret double %0
}
declare double @llvm.pow.f64(double, double) nounwind readonly

define double @test_acos() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_acos(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @acos(double 1.000000e+00)
  ret double %0
}

define double @test_asin() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_asin(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @asin(double 1.000000e+00)
  ret double %0
}

define double @test_atan() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_atan(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @atan(double 3.000000e+00)
  ret double %0
}

define double @test_atan2() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_atan2(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @atan2(double 3.000000e+00, double 4.000000e+00)
  ret double %0
}

define double @test_ceil() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_ceil(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @ceil(double 3.000000e+00)
  ret double %0
}

define double @test_cos() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_cos(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @cos(double 3.000000e+00)
  ret double %0
}

define double @test_cosh() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_cosh(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @cosh(double 3.000000e+00)
  ret double %0
}

define double @test_exp() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_exp(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @exp(double 3.000000e+00)
  ret double %0
}

define double @test_exp2() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_exp2(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @exp2(double 3.000000e+00)
  ret double %0
}

define double @test_fabs() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_fabs(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @fabs(double 3.000000e+00)
  ret double %0
}

define double @test_floor() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_floor(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @floor(double 3.000000e+00)
  ret double %0
}

define double @test_fmod() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_fmod(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @fmod(double 3.000000e+00, double 4.000000e+00)
  ret double %0
}

define double @test_log() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_log(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @log(double 3.000000e+00)
  ret double %0
}

define double @test_log10() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_log10(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @log10(double 3.000000e+00)
  ret double %0
}

define double @test_pow() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_pow(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @pow(double 3.000000e+00, double 4.000000e+00)
  ret double %0
}

define double @test_sin() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_sin(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @sin(double 3.000000e+00)
  ret double %0
}

define double @test_sinh() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_sinh(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @sinh(double 3.000000e+00)
  ret double %0
}

define double @test_sqrt() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_sqrt(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @sqrt(double 3.000000e+00)
  ret double %0
}

define double @test_tan() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_tan(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @tan(double 3.000000e+00)
  ret double %0
}

define double @test_tanh() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_tanh(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @tanh(double 3.000000e+00)
  ret double %0
}

define float @test_acosf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_acosf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @acosf(float 1.000000e+00)
  ret float %0
}

define float @test_asinf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_asinf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @asinf(float 1.000000e+00)
  ret float %0
}

define float @test_atanf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_atanf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @atanf(float 3.000000e+00)
  ret float %0
}

define float @test_atan2f() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_atan2f(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @atan2f(float 3.000000e+00, float 4.000000e+00)
  ret float %0
}

define float @test_ceilf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_ceilf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @ceilf(float 3.000000e+00)
  ret float %0
}

define float @test_cosf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_cosf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @cosf(float 3.000000e+00)
  ret float %0
}

define float @test_coshf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_coshf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @coshf(float 3.000000e+00)
  ret float %0
}

define float @test_expf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_expf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @expf(float 3.000000e+00)
  ret float %0
}

define float @test_exp2f() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_exp2f(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @exp2f(float 3.000000e+00)
  ret float %0
}

define float @test_fabsf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_fabsf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @fabsf(float 3.000000e+00)
  ret float %0
}

define float @test_floorf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_floorf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @floorf(float 3.000000e+00)
  ret float %0
}

define float @test_fmodf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_fmodf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @fmodf(float 3.000000e+00, float 4.000000e+00)
  ret float %0
}

define float @test_logf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_logf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @logf(float 3.000000e+00)
  ret float %0
}

define float @test_log10f() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_log10f(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @log10f(float 3.000000e+00)
  ret float %0
}

define float @test_powf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_powf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @powf(float 3.000000e+00, float 4.000000e+00)
  ret float %0
}

define float @test_sinf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_sinf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @sinf(float 3.000000e+00)
  ret float %0
}

define float @test_sinhf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_sinhf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @sinhf(float 3.000000e+00)
  ret float %0
}

define float @test_sqrtf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_sqrtf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @sqrtf(float 3.000000e+00)
  ret float %0
}

define float @test_tanf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_tanf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @tanf(float 3.000000e+00)
  ret float %0
}

define float @test_tanhf() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_tanhf(
; CHECK-NOT: call
; CHECK: ret
  %0 = call float @tanhf(float 3.000000e+00)
  ret float %0
}

; Shouldn't fold because of -fno-builtin
define double @acos_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @acos_(
; FNOBUILTIN: %1 = call double @acos(double 3.000000e+00)
  %1 = call double @acos(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @asin_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @asin_(
; FNOBUILTIN: %1 = call double @asin(double 3.000000e+00)
  %1 = call double @asin(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @atan_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @atan_(
; FNOBUILTIN: %1 = call double @atan(double 3.000000e+00)
  %1 = call double @atan(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @atan2_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @atan2_(
; FNOBUILTIN: %1 = call double @atan2(double 3.000000e+00, double 4.000000e+00)
  %1 = call double @atan2(double 3.000000e+00, double 4.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @ceil_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @ceil_(
; FNOBUILTIN: %1 = call double @ceil(double 3.000000e+00)
  %1 = call double @ceil(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @cos_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @cos_(
; FNOBUILTIN: %1 = call double @cos(double 3.000000e+00)
  %1 = call double @cos(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @cosh_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @cosh_(
; FNOBUILTIN: %1 = call double @cosh(double 3.000000e+00)
  %1 = call double @cosh(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @exp_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @exp_(
; FNOBUILTIN: %1 = call double @exp(double 3.000000e+00)
  %1 = call double @exp(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @exp2_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @exp2_(
; FNOBUILTIN: %1 = call double @exp2(double 3.000000e+00)
  %1 = call double @exp2(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @fabs_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @fabs_(
; FNOBUILTIN: %1 = call double @fabs(double 3.000000e+00)
  %1 = call double @fabs(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @floor_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @floor_(
; FNOBUILTIN: %1 = call double @floor(double 3.000000e+00)
  %1 = call double @floor(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @fmod_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @fmod_(
; FNOBUILTIN: %1 = call double @fmod(double 3.000000e+00, double 4.000000e+00)
  %1 = call double @fmod(double 3.000000e+00, double 4.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @log_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @log_(
; FNOBUILTIN: %1 = call double @log(double 3.000000e+00)
  %1 = call double @log(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @log10_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @log10_(
; FNOBUILTIN: %1 = call double @log10(double 3.000000e+00)
  %1 = call double @log10(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @pow_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @pow_(
; FNOBUILTIN: %1 = call double @pow(double 3.000000e+00, double 4.000000e+00)
  %1 = call double @pow(double 3.000000e+00, double 4.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @sin_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @sin_(
; FNOBUILTIN: %1 = call double @sin(double 3.000000e+00)
  %1 = call double @sin(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @sinh_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @sinh_(
; FNOBUILTIN: %1 = call double @sinh(double 3.000000e+00)
  %1 = call double @sinh(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @sqrt_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @sqrt_(
; FNOBUILTIN: %1 = call double @sqrt(double 3.000000e+00)
  %1 = call double @sqrt(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @tan_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @tan_(
; FNOBUILTIN: %1 = call double @tan(double 3.000000e+00)
  %1 = call double @tan(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @tanh_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @tanh_(
; FNOBUILTIN: %1 = call double @tanh(double 3.000000e+00)
  %1 = call double @tanh(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define float @acosf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @acosf_(
; FNOBUILTIN: %1 = call float @acosf(float 3.000000e+00)
  %1 = call float @acosf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @asinf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @asinf_(
; FNOBUILTIN: %1 = call float @asinf(float 3.000000e+00)
  %1 = call float @asinf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @atanf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @atanf_(
; FNOBUILTIN: %1 = call float @atanf(float 3.000000e+00)
  %1 = call float @atanf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @atan2f_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @atan2f_(
; FNOBUILTIN: %1 = call float @atan2f(float 3.000000e+00, float 4.000000e+00)
  %1 = call float @atan2f(float 3.000000e+00, float 4.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @ceilf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @ceilf_(
; FNOBUILTIN: %1 = call float @ceilf(float 3.000000e+00)
  %1 = call float @ceilf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @cosf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @cosf_(
; FNOBUILTIN: %1 = call float @cosf(float 3.000000e+00)
  %1 = call float @cosf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @coshf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @coshf_(
; FNOBUILTIN: %1 = call float @coshf(float 3.000000e+00)
  %1 = call float @coshf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @expf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @expf_(
; FNOBUILTIN: %1 = call float @expf(float 3.000000e+00)
  %1 = call float @expf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @exp2f_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @exp2f_(
; FNOBUILTIN: %1 = call float @exp2f(float 3.000000e+00)
  %1 = call float @exp2f(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @fabsf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @fabsf_(
; FNOBUILTIN: %1 = call float @fabsf(float 3.000000e+00)
  %1 = call float @fabsf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @floorf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @floorf_(
; FNOBUILTIN: %1 = call float @floorf(float 3.000000e+00)
  %1 = call float @floorf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @fmodf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @fmodf_(
; FNOBUILTIN: %1 = call float @fmodf(float 3.000000e+00, float 4.000000e+00)
  %1 = call float @fmodf(float 3.000000e+00, float 4.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @logf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @logf_(
; FNOBUILTIN: %1 = call float @logf(float 3.000000e+00)
  %1 = call float @logf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @log10f_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @log10f_(
; FNOBUILTIN: %1 = call float @log10f(float 3.000000e+00)
  %1 = call float @log10f(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @powf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @powf_(
; FNOBUILTIN: %1 = call float @powf(float 3.000000e+00, float 4.000000e+00)
  %1 = call float @powf(float 3.000000e+00, float 4.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @sinf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @sinf_(
; FNOBUILTIN: %1 = call float @sinf(float 3.000000e+00)
  %1 = call float @sinf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @sinhf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @sinhf_(
; FNOBUILTIN: %1 = call float @sinhf(float 3.000000e+00)
  %1 = call float @sinhf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @sqrtf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @sqrtf_(
; FNOBUILTIN: %1 = call float @sqrtf(float 3.000000e+00)
  %1 = call float @sqrtf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @tanf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @tanf_(
; FNOBUILTIN: %1 = call float @tanf(float 3.000000e+00)
  %1 = call float @tanf(float 3.000000e+00)
  ret float %1
}

; Shouldn't fold because of -fno-builtin
define float @tanhf_() nounwind uwtable ssp {
; FNOBUILTIN-LABEL: @tanhf_(
; FNOBUILTIN: %1 = call float @tanhf(float 3.000000e+00)
  %1 = call float @tanhf(float 3.000000e+00)
  ret float %1
}
