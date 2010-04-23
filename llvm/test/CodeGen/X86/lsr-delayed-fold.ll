; RUN: llc -march=x86-64 < %s > /dev/null
; rdar://7886751

; ScalarEvolution misses an opportunity to fold ((trunc x) + (trunc -x) + y),
; but LSR should tolerate this.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0"

define fastcc void @formatValue(i64 %arg5) nounwind {
bb12:                                             ; preds = %bb11
  %t = trunc i64 %arg5 to i32                   ; <i32> [#uses=1]
  %t13 = sub i64 0, %arg5                       ; <i64> [#uses=1]
  %t14 = and i64 %t13, 4294967295             ; <i64> [#uses=1]
  br label %bb15

bb15:                                             ; preds = %bb21, %bb12
  %t16 = phi i64 [ 0, %bb12 ], [ %t23, %bb15 ] ; <i64> [#uses=2]
  %t17 = mul i64 %t14, %t16                 ; <i64> [#uses=1]
  %t18 = add i64 undef, %t17                  ; <i64> [#uses=1]
  %t19 = trunc i64 %t18 to i32                ; <i32> [#uses=1]
  %t22 = icmp eq i32 %t19, %t               ; <i1> [#uses=1]
  %t23 = add i64 %t16, 1                      ; <i64> [#uses=1]
  br i1 %t22, label %bb24, label %bb15

bb24:                                             ; preds = %bb21, %bb11
  unreachable
}
