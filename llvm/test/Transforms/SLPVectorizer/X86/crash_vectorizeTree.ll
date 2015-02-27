; RUN: opt -basicaa -slp-vectorizer -mtriple=x86_64-apple-macosx10.9.0 -mcpu=corei7-avx -S < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"


; This test used to crash because we were following phi chains incorrectly.
; We used indices to get the incoming value of two phi nodes rather than 
; incoming block lookup.
; This can give wrong results when the ordering of incoming
; edges in the two phi nodes don't match.
;CHECK-LABEL: bar

%0 = type { %1, %2 }
%1 = type { double, double }
%2 = type { double, double }


;define fastcc void @bar() {
define void @bar() {
  %1 = getelementptr inbounds %0, %0* undef, i64 0, i32 1, i32 0
  %2 = getelementptr inbounds %0, %0* undef, i64 0, i32 1, i32 1
  %3 = getelementptr inbounds %0, %0* undef, i64 0, i32 1, i32 0
  %4 = getelementptr inbounds %0, %0* undef, i64 0, i32 1, i32 1
  %5 = getelementptr inbounds %0, %0* undef, i64 0, i32 1, i32 0
  %6 = getelementptr inbounds %0, %0* undef, i64 0, i32 1, i32 1
  br label %7

; <label>:7                                       ; preds = %18, %17, %17, %0
  %8 = phi double [ 2.800000e+01, %0 ], [ %11, %18 ], [ %11, %17 ], [ %11, %17 ]
  %9 = phi double [ 1.800000e+01, %0 ], [ %10, %18 ], [ %10, %17 ], [ %10, %17 ]
  store double %9, double* %1, align 8
  store double %8, double* %2, align 8
  %10 = load double, double* %3, align 8
  %11 = load double, double* %4, align 8
  br i1 undef, label %12, label %13

; <label>:12                                      ; preds = %7
  ret void

; <label>:13                                      ; preds = %7
  store double %10, double* %5, align 8
  store double %11, double* %6, align 8
  br i1 undef, label %14, label %15

; <label>:14                                      ; preds = %13
  br label %15

; <label>:15                                      ; preds = %14, %13
  br i1 undef, label %16, label %17

; <label>:16                                      ; preds = %15
  unreachable

; <label>:17                                      ; preds = %15
  switch i32 undef, label %18 [
    i32 32, label %7
    i32 103, label %7
  ]

; <label>:18                                      ; preds = %17
  br i1 undef, label %7, label %19

; <label>:19                                      ; preds = %18
  unreachable
}
