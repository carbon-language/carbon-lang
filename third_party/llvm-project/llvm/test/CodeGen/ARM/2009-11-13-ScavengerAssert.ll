; RUN: llc -mtriple=armv7-eabi -mcpu=cortex-a8 < %s
; PR5411

%bar = type { %quad, float, float, [3 x %quux*], [3 x %bar*], [2 x %bar*], [3 x i8], i8 }
%baz = type { %bar*, i32 }
%foo = type { i8, %quuz, %quad, float, [64 x %quux], [128 x %bar], i32, %baz, %baz }
%quad = type { [4 x float] }
%quux = type { %quad, %quad }
%quuz = type { [4 x %quux*], [4 x float], i32 }

define arm_aapcs_vfpcc %bar* @aaa(%foo* nocapture %this, %quux* %a, %quux* %b, %quux* %c, i8 zeroext %forced) {
entry:
  br i1 undef, label %bb85, label %bb

bb:                                               ; preds = %entry
  %0 = getelementptr inbounds %bar, %bar* null, i32 0, i32 0, i32 0, i32 2 ; <float*> [#uses=2]
  %1 = load float, float* undef, align 4                 ; <float> [#uses=1]
  %2 = fsub float 0.000000e+00, undef             ; <float> [#uses=2]
  %3 = fmul float 0.000000e+00, undef             ; <float> [#uses=1]
  %4 = load float, float* %0, align 4                    ; <float> [#uses=3]
  %5 = fmul float %4, %2                          ; <float> [#uses=1]
  %6 = fsub float %3, %5                          ; <float> [#uses=1]
  %7 = fmul float %4, undef                       ; <float> [#uses=1]
  %8 = fsub float %7, undef                       ; <float> [#uses=1]
  %9 = fmul float undef, %2                       ; <float> [#uses=1]
  %10 = fmul float 0.000000e+00, undef            ; <float> [#uses=1]
  %11 = fsub float %9, %10                        ; <float> [#uses=1]
  %12 = fmul float undef, %6                      ; <float> [#uses=1]
  %13 = fmul float 0.000000e+00, %8               ; <float> [#uses=1]
  %14 = fadd float %12, %13                       ; <float> [#uses=1]
  %15 = fmul float %1, %11                        ; <float> [#uses=1]
  %16 = fadd float %14, %15                       ; <float> [#uses=1]
  %17 = select i1 undef, float undef, float %16   ; <float> [#uses=1]
  %18 = fdiv float %17, 0.000000e+00              ; <float> [#uses=1]
  store float %18, float* undef, align 4
  %19 = fmul float %4, undef                      ; <float> [#uses=1]
  store float %19, float* %0, align 4
  ret %bar* null

bb85:                                             ; preds = %entry
  ret %bar* null
}
