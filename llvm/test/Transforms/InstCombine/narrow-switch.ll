; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"

; CHECK-LABEL: define i32 @positive1
; CHECK: switch i32
; CHECK: i32 10, label
; CHECK: i32 100, label
; CHECK: i32 1001, label

define i32 @positive1(i64 %a) {
entry:
  %and = and i64 %a, 4294967295
  switch i64 %and, label %sw.default [
    i64 10, label %return
    i64 100, label %sw.bb1
    i64 1001, label %sw.bb2
  ]

sw.bb1:
  br label %return

sw.bb2:
  br label %return

sw.default:
  br label %return

return:
  %retval.0 = phi i32 [ 24, %sw.default ], [ 123, %sw.bb2 ], [ 213, %sw.bb1 ], [ 231, %entry ]
  ret i32 %retval.0
}

; CHECK-LABEL: define i32 @negative1
; CHECK: switch i32
; CHECK: i32 -10, label
; CHECK: i32 -100, label
; CHECK: i32 -1001, label

define i32 @negative1(i64 %a) {
entry:
  %or = or i64 %a, -4294967296
  switch i64 %or, label %sw.default [
    i64 -10, label %return
    i64 -100, label %sw.bb1
    i64 -1001, label %sw.bb2
  ]

sw.bb1:
  br label %return

sw.bb2:
  br label %return

sw.default:
  br label %return

return:
  %retval.0 = phi i32 [ 24, %sw.default ], [ 123, %sw.bb2 ], [ 213, %sw.bb1 ], [ 231, %entry ]
  ret i32 %retval.0
}

; Make sure truncating a constant int larger than 64-bit doesn't trigger an
; assertion.

; CHECK-LABEL: define i32 @trunc72to68
; CHECK: switch i68
; CHECK: i68 10, label
; CHECK: i68 100, label
; CHECK: i68 1001, label

define i32 @trunc72to68(i72 %a) {
entry:
  %and = and i72 %a, 295147905179352825855
  switch i72 %and, label %sw.default [
    i72 10, label %return
    i72 100, label %sw.bb1
    i72 1001, label %sw.bb2
  ]

sw.bb1:
  br label %return

sw.bb2:
  br label %return

sw.default:
  br label %return

return:
  %retval.0 = phi i32 [ 24, %sw.default ], [ 123, %sw.bb2 ], [ 213, %sw.bb1 ], [ 231, %entry ]
  ret i32 %retval.0
}
