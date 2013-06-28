; ModuleID = 'function.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind uwtable
define void @blah(i32* %i) #0 {
  %1 = alloca i32*, align 8                   ; CHECK: !dbg
  store i32* %i, i32** %1, align 8            ; CHECK-NEXT: !dbg
  %2 = load i32** %1, align 8                 ; CHECK-NEXT: !dbg
  %3 = load i32* %2, align 4                  ; CHECK-NEXT: !dbg
  %4 = add nsw i32 %3, 1                      ; CHECK-NEXT: !dbg
  store i32 %4, i32* %2, align 4              ; CHECK-NEXT: !dbg
  ret void                                    ; CHECK-NEXT: !dbg
}

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** %argv) #0 {
  %1 = alloca i32, align 4                    ; CHECK: !dbg
  %2 = alloca i32, align 4                    ; CHECK-NEXT: !dbg
  %3 = alloca i8**, align 8                   ; CHECK-NEXT: !dbg
  %i = alloca i32, align 4                    ; CHECK-NEXT: !dbg
  store i32 0, i32* %1                        ; CHECK-NEXT: !dbg
  store i32 %argc, i32* %2, align 4           ; CHECK-NEXT: !dbg
  store i8** %argv, i8*** %3, align 8         ; CHECK-NEXT: !dbg
  store i32 7, i32* %i, align 4               ; CHECK-NEXT: !dbg
  call void @blah(i32* %i)                    ; CHECK-NEXT: !dbg
  %4 = load i32* %i, align 4                  ; CHECK-NEXT: !dbg
  ret i32 %4                                  ; CHECK-NEXT: !dbg
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
; CHECK: = metadata !{i32 8,
; CHECK-NEXT: = metadata !{i32 9,
; CHECK-NEXT: = metadata !{i32 10,
; CHECK-NEXT: = metadata !{i32 11,
; CHECK-NEXT: = metadata !{i32 12,
; CHECK-NEXT: = metadata !{i32 13,

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

; RUN: opt %s -debug-ir -S | FileCheck %s
