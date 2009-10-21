; RUN: opt -instcombine -S <%s | FileCheck %s
; PR5262
; Make sure the PHI node gets put in a place where all of its operands dominate it
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@tmp2 = global i64 0                              ; <i64*> [#uses=1]

declare void @use(i64) nounwind

define void @foo(i1) nounwind align 2 {
; <label>:1
  br i1 %0, label %2, label %3

; <label>:2                                       ; preds = %1
  br label %3

; <label>:3                                       ; preds = %2, %1
; CHECK: <label>:3
; CHECK-NEXT: %v4 = phi i1 [ false, %2 ], [ true, %1 ]
; CHECK-NEXT: %v4_ =  phi i1 [ true, %2 ], [ false, %1 ]
; CHECK-NEXT: br label %l8
  %v4 = phi i8 [ 1, %2 ], [ 0, %1 ]                ; <i8> [#uses=1]
  %v4_ = phi i8 [ 0, %2 ], [ 1, %1 ]                ; <i8> [#uses=1]
  %v5 = icmp eq i8 %v4, 0                           ; <i1> [#uses=1]
  %v5_ = icmp eq i8 %v4_, 0                           ; <i1> [#uses=1]
  %v6 = load i64* @tmp2, align 8                   ; <i64> [#uses=1]
  %v7 = select i1 %v5, i64 0, i64 %v6                ; <i64> [#uses=1]
  br label %l8

l8:
  %v9 = load i64* @tmp2, align 8
  call void @use(i64 %v7)
  br label %l10
l10:
  %v11 = select i1 %v5_, i64 0, i64 %v9                ; <i64> [#uses=1]
  call void @use(i64 %v11)
  br label %l11
l11:
  %v12 = load i64* @tmp2, align 8
  %v13 = select i1 %v5_, i64 0, i64 %v12                ; <i64> [#uses=1]
  call void @use(i64 %v13)
  br i1 %0, label %l12, label %l13
l12:
  br label %l13
l13:
;CHECK: l13:
;CHECK-NEXT: %v14 = phi i64 [ %v12, %l12 ], [ 0, %l11 ]
;CHECK-NEXT: call void @use(i64 %v14)
  %v14 = phi i1 [0, %l12], [1, %l11]
  %v16 = select i1 %v14, i64 0, i64 %v12                ; <i64> [#uses=1]
  call void @use(i64 %v16)
  ret void
}
