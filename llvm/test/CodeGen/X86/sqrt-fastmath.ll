; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=core2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=btver2 | FileCheck %s --check-prefix=BTVER2

; generated using "clang -S -O2 -ffast-math -emit-llvm sqrt.c" from
; #include <math.h>
; 
; double fd(double d){
;   return sqrt(d);
; }
; 
; float ff(float f){
;   return sqrtf(f);
; }
; 
; long double fld(long double ld){
;   return sqrtl(ld);
; }
;
; Tests conversion of sqrt function calls into sqrt instructions when
; -ffast-math is in effect.

; ModuleID = 'sqrt.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define double @fd(double %d) #0 {
entry:
; CHECK: sqrtsd
  %call = tail call double @__sqrt_finite(double %d) #2
  ret double %call
}

; Function Attrs: nounwind readnone
declare double @__sqrt_finite(double) #1

; Function Attrs: nounwind readnone uwtable
define float @ff(float %f) #0 {
entry:
; CHECK: sqrtss
  %call = tail call float @__sqrtf_finite(float %f) #2
  ret float %call
}

; Function Attrs: nounwind readnone
declare float @__sqrtf_finite(float) #1

; Function Attrs: nounwind readnone uwtable
define x86_fp80 @fld(x86_fp80 %ld) #0 {
entry:
; CHECK: fsqrt
  %call = tail call x86_fp80 @__sqrtl_finite(x86_fp80 %ld) #2
  ret x86_fp80 %call
}

declare x86_fp80 @__sqrtl_finite(x86_fp80) #1

declare float @llvm.sqrt.f32(float) #1
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) #1
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #1

; If the target's sqrtss and divss instructions are substantially
; slower than rsqrtss with a Newton-Raphson refinement, we should
; generate the estimate sequence.

define float @reciprocal_square_root(float %x) #0 {
  %sqrt = tail call float @llvm.sqrt.f32(float %x)
  %div = fdiv fast float 1.0, %sqrt
  ret float %div

; CHECK-LABEL: reciprocal_square_root:
; CHECK: sqrtss
; CHECK-NEXT: movss
; CHECK-NEXT: divss
; CHECK-NEXT: retq
; BTVER2-LABEL: reciprocal_square_root:
; BTVER2: vrsqrtss
; BTVER2-NEXT: vmulss
; BTVER2-NEXT: vmulss
; BTVER2-NEXT: vmulss
; BTVER2-NEXT: vaddss
; BTVER2-NEXT: vmulss
; BTVER2-NEXT: retq
}

define <4 x float> @reciprocal_square_root_v4f32(<4 x float> %x) #0 {
  %sqrt = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %div = fdiv fast <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %sqrt
  ret <4 x float> %div

; CHECK-LABEL: reciprocal_square_root_v4f32:
; CHECK: sqrtps
; CHECK-NEXT: movaps
; CHECK-NEXT: divps
; CHECK-NEXT: retq
; BTVER2-LABEL: reciprocal_square_root_v4f32:
; BTVER2: vrsqrtps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: vaddps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: retq
}

define <8 x float> @reciprocal_square_root_v8f32(<8 x float> %x) #0 {
  %sqrt = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %x)
  %div = fdiv fast <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, %sqrt
  ret <8 x float> %div

; CHECK-LABEL: reciprocal_square_root_v8f32:
; CHECK: sqrtps
; CHECK-NEXT: sqrtps
; CHECK-NEXT: movaps
; CHECK-NEXT: movaps
; CHECK-NEXT: divps
; CHECK-NEXT: divps
; CHECK-NEXT: retq
; BTVER2-LABEL: reciprocal_square_root_v8f32:
; BTVER2: vrsqrtps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: vaddps
; BTVER2-NEXT: vmulps
; BTVER2-NEXT: retq
}


attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
