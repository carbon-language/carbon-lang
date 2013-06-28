; ModuleID = 'vector.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define <4 x float> @_Z3fooDv2_fS_(double %a.coerce, double %b.coerce) #0 {
  %1 = alloca <2 x float>, align 8                    ; CHECK: !dbg
  %2 = alloca <2 x float>, align 8                    ; CHECK-NEXT: !dbg
  %3 = alloca <2 x float>, align 8                    ; CHECK-NEXT: !dbg
  %4 = alloca <2 x float>, align 8                    ; CHECK-NEXT: !dbg
  %c = alloca <4 x float>, align 16                   ; CHECK-NEXT: !dbg
  %5 = bitcast <2 x float>* %1 to double*             ; CHECK-NEXT: !dbg
  store double %a.coerce, double* %5, align 1         ; CHECK-NEXT: !dbg
  %a = load <2 x float>* %1, align 8                  ; CHECK-NEXT: !dbg
  store <2 x float> %a, <2 x float>* %2, align 8      ; CHECK-NEXT: !dbg
  %6 = bitcast <2 x float>* %3 to double*             ; CHECK-NEXT: !dbg
  store double %b.coerce, double* %6, align 1         ; CHECK-NEXT: !dbg
  %b = load <2 x float>* %3, align 8                  ; CHECK-NEXT: !dbg
  store <2 x float> %b, <2 x float>* %4, align 8      ; CHECK-NEXT: !dbg
  %7 = load <2 x float>* %2, align 8                  ; CHECK-NEXT: !dbg
  %8 = load <4 x float>* %c, align 16                 ; CHECK-NEXT: !dbg
  %9 = shufflevector <2 x float> %7, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>   ; CHECK-NEXT: !dbg
  %10 = shufflevector <4 x float> %8, <4 x float> %9, <4 x i32> <i32 4, i32 1, i32 5, i32 3>             ; CHECK-NEXT: !dbg
  store <4 x float> %10, <4 x float>* %c, align 16    ; CHECK-NEXT: !dbg
  %11 = load <2 x float>* %4, align 8                 ; CHECK-NEXT: !dbg
  %12 = load <4 x float>* %c, align 16                ; CHECK-NEXT: !dbg
  %13 = shufflevector <2 x float> %11, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef> ; CHECK-NEXT: !dbg
  %14 = shufflevector <4 x float> %12, <4 x float> %13, <4 x i32> <i32 0, i32 4, i32 2, i32 5>           ; CHECK-NEXT: !dbg
  store <4 x float> %14, <4 x float>* %c, align 16    ; CHECK-NEXT: !dbg
  %15 = load <4 x float>* %c, align 16                ; CHECK-NEXT: !dbg
  ret <4 x float> %15                                 ; CHECK-NEXT: !dbg
}

; Function Attrs: nounwind uwtable
define i32 @main() #1 {
  %1 = alloca i32, align 4                            ; CHECK: !dbg
  %a = alloca <2 x float>, align 8                    ; CHECK-NEXT: !dbg
  %b = alloca <2 x float>, align 8                    ; CHECK-NEXT: !dbg
  %x = alloca <4 x float>, align 16                   ; CHECK-NEXT: !dbg
  %2 = alloca <2 x float>, align 8                    ; CHECK-NEXT: !dbg
  %3 = alloca <2 x float>, align 8                    ; CHECK-NEXT: !dbg
  store i32 0, i32* %1                                ; CHECK-NEXT: !dbg
  store <2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float>* %a, align 8                   ; CHECK-NEXT: !dbg
  store <2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float>* %b, align 8                   ; CHECK-NEXT: !dbg
  %4 = load <2 x float>* %a, align 8                  ; CHECK-NEXT: !dbg
  %5 = load <2 x float>* %b, align 8                  ; CHECK-NEXT: !dbg
  store <2 x float> %4, <2 x float>* %2, align 8      ; CHECK-NEXT: !dbg
  %6 = bitcast <2 x float>* %2 to double*             ; CHECK-NEXT: !dbg
  %7 = load double* %6, align 1                       ; CHECK-NEXT: !dbg
  store <2 x float> %5, <2 x float>* %3, align 8      ; CHECK-NEXT: !dbg
  %8 = bitcast <2 x float>* %3 to double*             ; CHECK-NEXT: !dbg
  %9 = load double* %8, align 1                       ; CHECK-NEXT: !dbg
  %10 = call <4 x float> @_Z3fooDv2_fS_(double %7, double %9)                                            ; CHECK-NEXT: !dbg
  store <4 x float> %10, <4 x float>* %x, align 16    ; CHECK-NEXT: !dbg
  %11 = load <4 x float>* %x, align 16                ; CHECK-NEXT: !dbg
  %12 = extractelement <4 x float> %11, i32 0         ; CHECK-NEXT: !dbg
  %13 = load <4 x float>* %x, align 16                ; CHECK-NEXT: !dbg
  %14 = extractelement <4 x float> %13, i32 1         ; CHECK-NEXT: !dbg
  %15 = fadd float %12, %14                           ; CHECK-NEXT: !dbg
  %16 = load <4 x float>* %x, align 16                ; CHECK-NEXT: !dbg
  %17 = extractelement <4 x float> %16, i32 2         ; CHECK-NEXT: !dbg
  %18 = fadd float %15, %17                           ; CHECK-NEXT: !dbg
  %19 = load <4 x float>* %x, align 16                ; CHECK-NEXT: !dbg
  %20 = extractelement <4 x float> %19, i32 3         ; CHECK-NEXT: !dbg
  %21 = fadd float %18, %20                           ; CHECK-NEXT: !dbg
  %22 = fptosi float %21 to i32                       ; CHECK-NEXT: !dbg
  ret i32 %22                                         ; CHECK-NEXT: !dbg
}

attributes #0 = { noinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

; CHECK: = metadata !{i32 13,
; CHECK-NEXT: = metadata !{i32 14,
; CHECK-NEXT: = metadata !{i32 15,
; CHECK-NEXT: = metadata !{i32 16,
; CHECK-NEXT: = metadata !{i32 17,
; CHECK-NEXT: = metadata !{i32 18,
; CHECK-NEXT: = metadata !{i32 19,
; CHECK-NEXT: = metadata !{i32 20,
; CHECK-NEXT: = metadata !{i32 21,
; CHECK-NEXT: = metadata !{i32 22,
; CHECK-NEXT: = metadata !{i32 23,
; CHECK-NEXT: = metadata !{i32 24,
; CHECK-NEXT: = metadata !{i32 25,
; CHECK-NEXT: = metadata !{i32 26,
; CHECK-NEXT: = metadata !{i32 27,
; CHECK-NEXT: = metadata !{i32 28,
; CHECK-NEXT: = metadata !{i32 29,
; CHECK-NEXT: = metadata !{i32 30,
; CHECK-NEXT: = metadata !{i32 31,

; RUN: opt %s -debug-ir -S | FileCheck %s
