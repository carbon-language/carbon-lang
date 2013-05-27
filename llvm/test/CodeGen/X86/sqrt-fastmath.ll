; RUN: llc < %s -mcpu=core2 | FileCheck %s

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

; Function Attrs: nounwind readnone
declare x86_fp80 @__sqrtl_finite(x86_fp80) #1

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
