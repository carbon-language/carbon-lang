; ModuleID = 'crash.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [18 x i8] c"Hello, segfault!\0A\00", align 1
@.str1 = private unnamed_addr constant [14 x i8] c"Now crash %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** %argv) #0 {
  %1 = alloca i32, align 4                                        ;CHECK: !dbg
  %2 = alloca i32, align 4                                        ;CHECK-NEXT: !dbg
  %3 = alloca i8**, align 8                                       ;CHECK-NEXT: !dbg
  %null_ptr = alloca i32*, align 8                                ;CHECK-NEXT: !dbg
  store i32 0, i32* %1                                            ;CHECK-NEXT: !dbg
  store i32 %argc, i32* %2, align 4                               ;CHECK-NEXT: !dbg
  store i8** %argv, i8*** %3, align 8                             ;CHECK-NEXT: !dbg
  store i32* null, i32** %null_ptr, align 8                       ;CHECK-NEXT: !dbg
  %4 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([18 x i8]* @.str, i32 0, i32 0)) ;CHECK-NEXT: !dbg
  %5 = load i32** %null_ptr, align 8                              ;CHECK-NEXT: !dbg
  %6 = load i32* %5, align 4                                      ;CHECK-NEXT: !dbg
  %7 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([14 x i8]* @.str1, i32 0, i32 0), i32 %6) ;CHECK-NEXT: !dbg
  %8 = load i32* %2, align 4                                      ;CHECK-NEXT: !dbg
  ret i32 %8                                                      ;CHECK-NEXT: !dbg
}

declare i32 @printf(i8*, ...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

; CHECK: = metadata !{i32 14,
; CHECK-NEXT: = metadata !{i32 15,
; CHECK-NEXT: = metadata !{i32 16,
; CHECK-NEXT: = metadata !{i32 17,
; CHECK-NEXT: = metadata !{i32 18,
; CHECK-NEXT: = metadata !{i32 19,
; CHECK-NEXT: = metadata !{i32 20,
; CHECK-NEXT: = metadata !{i32 21,
; CHECK-NEXT: = metadata !{i32 22,
; CHECK-NEXT: = metadata !{i32 23,

; RUN: opt %s -debug-ir -S | FileCheck %s
