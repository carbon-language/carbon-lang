; ModuleID = 'exception.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@_ZTIi = external constant i8*

; Function Attrs: uwtable
define i32 @main(i32 %argc, i8** %argv) #0 {
  %1 = alloca i32, align 4                        ; CHECK: !dbg
  %2 = alloca i32, align 4                        ; CHECK-NEXT: !dbg
  %3 = alloca i8**, align 8                       ; CHECK-NEXT: !dbg
  %4 = alloca i8*                                 ; CHECK-NEXT: !dbg
  %5 = alloca i32                                 ; CHECK-NEXT: !dbg
  %e = alloca i32, align 4                        ; CHECK-NEXT: !dbg
  %6 = alloca i32                                 ; CHECK-NEXT: !dbg
  store i32 0, i32* %1                            ; CHECK-NEXT: !dbg
  store i32 %argc, i32* %2, align 4               ; CHECK-NEXT: !dbg
  store i8** %argv, i8*** %3, align 8             ; CHECK-NEXT: !dbg
  %7 = call i8* @__cxa_allocate_exception(i64 4) #2 ; CHECK-NEXT: !dbg
  %8 = bitcast i8* %7 to i32*                     ; CHECK-NEXT: !dbg
  %9 = load i32* %2, align 4                      ; CHECK-NEXT: !dbg
  store i32 %9, i32* %8                           ; CHECK-NEXT: !dbg
  invoke void @__cxa_throw(i8* %7, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #3
          to label %31 unwind label %10           ; CHECK: !dbg

; <label>:10                                      ; preds = %0
  %11 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)  ; CHECK: !dbg
  %12 = extractvalue { i8*, i32 } %11, 0          ; CHECK-NEXT: !dbg
  store i8* %12, i8** %4                          ; CHECK-NEXT: !dbg
  %13 = extractvalue { i8*, i32 } %11, 1          ; CHECK-NEXT: !dbg
  store i32 %13, i32* %5                          ; CHECK-NEXT: !dbg
  br label %14                                    ; CHECK-NEXT: !dbg

; <label>:14                                      ; preds = %10
  %15 = load i32* %5                              ; CHECK: !dbg
  %16 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2   ; CHECK-NEXT: !dbg
  %17 = icmp eq i32 %15, %16                      ; CHECK-NEXT: !dbg
  br i1 %17, label %18, label %26                 ; CHECK-NEXT: !dbg

; <label>:18                                      ; preds = %14
  %19 = load i8** %4                              ; CHECK: !dbg
  %20 = call i8* @__cxa_begin_catch(i8* %19) #2   ; CHECK-NEXT: !dbg
  %21 = bitcast i8* %20 to i32*                   ; CHECK-NEXT: !dbg
  %22 = load i32* %21, align 4                    ; CHECK-NEXT: !dbg
  store i32 %22, i32* %e, align 4                 ; CHECK-NEXT: !dbg
  %23 = load i32* %e, align 4                     ; CHECK-NEXT: !dbg
  store i32 %23, i32* %1                          ; CHECK-NEXT: !dbg
  store i32 1, i32* %6                            ; CHECK-NEXT: !dbg
  call void @__cxa_end_catch() #2                 ; CHECK-NEXT: !dbg
  br label %24                                    ; CHECK-NEXT: !dbg

; <label>:24                                      ; preds = %18
  %25 = load i32* %1                              ; CHECK: !dbg
  ret i32 %25                                     ; CHECK-NEXT: !dbg

; <label>:26                                      ; preds = %14
  %27 = load i8** %4                              ; CHECK: !dbg
  %28 = load i32* %5                              ; CHECK-NEXT: !dbg
  %29 = insertvalue { i8*, i32 } undef, i8* %27, 0 ; CHECK-NEXT: !dbg
  %30 = insertvalue { i8*, i32 } %29, i32 %28, 1   ; CHECK-NEXT: !dbg
  resume { i8*, i32 } %30                         ; CHECK-NEXT: !dbg

; <label>:31                                      ; preds = %0
  unreachable                                     ; CHECK: !dbg
}

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn }
; CHECK: = metadata !{i32 16,
; CHECK-NEXT: = metadata !{i32 17,
; CHECK-NEXT: = metadata !{i32 18,
; CHECK-NEXT: = metadata !{i32 19,
; CHECK-NEXT: = metadata !{i32 20,
; CHECK-NEXT: = metadata !{i32 21,
; CHECK-NEXT: = metadata !{i32 22,
; CHECK-NEXT: = metadata !{i32 24,

; CHECK-NEXT: = metadata !{i32 28,
; CHECK-NEXT: = metadata !{i32 29,
; CHECK-NEXT: = metadata !{i32 30,
; CHECK-NEXT: = metadata !{i32 31,
; CHECK-NEXT: = metadata !{i32 32,
; CHECK-NEXT: = metadata !{i32 33,

; CHECK-NEXT: = metadata !{i32 36,
; CHECK-NEXT: = metadata !{i32 37,
; CHECK-NEXT: = metadata !{i32 38,
; CHECK-NEXT: = metadata !{i32 39,

; CHECK-NEXT: = metadata !{i32 42,
; CHECK-NEXT: = metadata !{i32 43,
; CHECK-NEXT: = metadata !{i32 44,
; CHECK-NEXT: = metadata !{i32 45,
; CHECK-NEXT: = metadata !{i32 46,
; CHECK-NEXT: = metadata !{i32 47,
; CHECK-NEXT: = metadata !{i32 48,
; CHECK-NEXT: = metadata !{i32 49,
; CHECK-NEXT: = metadata !{i32 50,
; CHECK-NEXT: = metadata !{i32 51,

; CHECK-NEXT: = metadata !{i32 54,
; CHECK-NEXT: = metadata !{i32 55,

; CHECK-NEXT: = metadata !{i32 58,
; CHECK-NEXT: = metadata !{i32 59,
; CHECK-NEXT: = metadata !{i32 60,
; CHECK-NEXT: = metadata !{i32 61,
; CHECK-NEXT: = metadata !{i32 62,
; CHECK-NEXT: = metadata !{i32 65,

; RUN: opt %s -debug-ir -S | FileCheck %s
