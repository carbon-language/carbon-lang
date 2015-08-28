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
; FNOBUILTIN-LABEL: @T(

; CHECK-NOT: call
; CHECK: ret
  %A = call double @cos(double 0.000000e+00)
  %B = call double @sin(double 0.000000e+00)
  %a = fadd double %A, %B
  %C = call double @tan(double 0.000000e+00)
  %b = fadd double %a, %C
  %D = call double @sqrt(double 4.000000e+00)
  %c = fadd double %b, %D

  %slot = alloca double
  %slotf = alloca float
; FNOBUILTIN: call
  %1 = call double @acos(double 1.000000e+00)
  store double %1, double* %slot
; FNOBUILTIN: call
  %2 = call double @asin(double 1.000000e+00)
  store double %2, double* %slot
; FNOBUILTIN: call
  %3 = call double @atan(double 3.000000e+00)
  store double %3, double* %slot
; FNOBUILTIN: call
  %4 = call double @atan2(double 3.000000e+00, double 4.000000e+00)
  store double %4, double* %slot
; FNOBUILTIN: call
  %5 = call double @ceil(double 3.000000e+00)
  store double %5, double* %slot
; FNOBUILTIN: call
  %6 = call double @cosh(double 3.000000e+00)
  store double %6, double* %slot
; FNOBUILTIN: call
  %7 = call double @exp(double 3.000000e+00)
  store double %7, double* %slot
; FNOBUILTIN: call
  %8 = call double @exp2(double 3.000000e+00)
  store double %8, double* %slot
; FNOBUILTIN: call
  %9 = call double @fabs(double 3.000000e+00)
  store double %9, double* %slot
; FNOBUILTIN: call
  %10 = call double @floor(double 3.000000e+00)
  store double %10, double* %slot
; FNOBUILTIN: call
  %11 = call double @fmod(double 3.000000e+00, double 4.000000e+00)
  store double %11, double* %slot
; FNOBUILTIN: call
  %12 = call double @log(double 3.000000e+00)
  store double %12, double* %slot
; FNOBUILTIN: call
  %13 = call double @log10(double 3.000000e+00)
  store double %13, double* %slot
; FNOBUILTIN: call
  %14 = call double @pow(double 3.000000e+00, double 4.000000e+00)
  store double %14, double* %slot
; FNOBUILTIN: call
  %15 = call double @sinh(double 3.000000e+00)
  store double %15, double* %slot
; FNOBUILTIN: call
  %16 = call double @tanh(double 3.000000e+00)
  store double %16, double* %slot
; FNOBUILTIN: call
  %17 = call float @acosf(float 1.000000e+00)
  store float %17, float* %slotf
; FNOBUILTIN: call
  %18 = call float @asinf(float 1.000000e+00)
  store float %18, float* %slotf
; FNOBUILTIN: call
  %19 = call float @atanf(float 3.000000e+00)
  store float %19, float* %slotf
; FNOBUILTIN: call
  %20 = call float @atan2f(float 3.000000e+00, float 4.000000e+00)
  store float %20, float* %slotf
; FNOBUILTIN: call
  %21 = call float @ceilf(float 3.000000e+00)
  store float %21, float* %slotf
; FNOBUILTIN: call
  %22 = call float @cosf(float 3.000000e+00)
  store float %22, float* %slotf
; FNOBUILTIN: call
  %23 = call float @coshf(float 3.000000e+00)
  store float %23, float* %slotf
; FNOBUILTIN: call
  %24 = call float @expf(float 3.000000e+00)
  store float %24, float* %slotf
; FNOBUILTIN: call
  %25 = call float @exp2f(float 3.000000e+00)
  store float %25, float* %slotf
; FNOBUILTIN: call
  %26 = call float @fabsf(float 3.000000e+00)
  store float %26, float* %slotf
; FNOBUILTIN: call
  %27 = call float @floorf(float 3.000000e+00)
  store float %27, float* %slotf
; FNOBUILTIN: call
  %28 = call float @fmodf(float 3.000000e+00, float 4.000000e+00)
  store float %28, float* %slotf
; FNOBUILTIN: call
  %29 = call float @logf(float 3.000000e+00)
  store float %29, float* %slotf
; FNOBUILTIN: call
  %30 = call float @log10f(float 3.000000e+00)
  store float %30, float* %slotf
; FNOBUILTIN: call
  %31 = call float @powf(float 3.000000e+00, float 4.000000e+00)
  store float %31, float* %slotf
; FNOBUILTIN: call
  %32 = call float @sinf(float 3.000000e+00)
  store float %32, float* %slotf
; FNOBUILTIN: call
  %33 = call float @sinhf(float 3.000000e+00)
  store float %33, float* %slotf
; FNOBUILTIN: call
  %34 = call float @sqrtf(float 3.000000e+00)
  store float %34, float* %slotf
; FNOBUILTIN: call
  %35 = call float @tanf(float 3.000000e+00)
  store float %35, float* %slotf
; FNOBUILTIN: call
  %36 = call float @tanhf(float 3.000000e+00)
  store float %36, float* %slotf

; FNOBUILTIN: ret

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
