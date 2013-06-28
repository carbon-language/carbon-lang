; ModuleID = 'struct.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.blah = type { i32, float, i8 }

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
  %1 = alloca i32, align 4                                    ; CHECK: !dbg
  %b = alloca %struct.blah, align 4                           ; CHECK-NEXT: !dbg
  store i32 0, i32* %1                                        ; CHECK-NEXT: !dbg
  %2 = getelementptr inbounds %struct.blah* %b, i32 0, i32 0  ; CHECK-NEXT: !dbg
  %3 = load i32* %2, align 4                                  ; CHECK-NEXT: !dbg
  ret i32 %3                                                  ; CHECK-NEXT: !dbg
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

; CHECK: = metadata !{i32 11,
; CHECK-NEXT: = metadata !{i32 12,
; CHECK-NEXT: = metadata !{i32 13,
; CHECK-NEXT: = metadata !{i32 14,

; RUN: opt %s -debug-ir -S | FileCheck %s
