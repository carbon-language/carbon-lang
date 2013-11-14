; RUN: opt < %s -msan -msan-check-access-address=0 -msan-wrap-indirect-calls=zzz -msan-wrap-indirect-calls-fast=0 -S | FileCheck %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-wrap-indirect-calls=zzz -msan-wrap-indirect-calls-fast=1 -S | FileCheck -check-prefix=CHECK-FAST %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Test for -msan-wrap-indirect-calls functionality.
; Replaces indirect call to %f with a call to whatever is returned from the
; wrapper function.

; This does not depend on the sanitize_memory attribute.
define i32 @func(i32 (i32, i32)* nocapture %f, i32 %x, i32 %y) {
entry:
  %call = tail call i32 %f(i32 %x, i32 %y)
  ret i32 %call
}

; CHECK: @func
; CHECK: bitcast i32 (i32, i32)* %f to void ()*
; CHECK: call void ()* (void ()*)* @zzz(void ()*
; CHECK: [[A:%[01-9a-z_.]+]] = bitcast void ()* {{.*}} to i32 (i32, i32)*
; CHECK: call i32 {{.*}}[[A]](i32 {{.*}}, i32 {{.*}})
; CHECK: ret i32

; CHECK-FAST: @func
; CHECK-FAST: bitcast i32 (i32, i32)* %f to void ()*
; CHECK-FAST-DAG: icmp ult void ()* {{.*}}, bitcast (i32* @__executable_start to void ()*)
; CHECK-FAST-DAG: icmp uge void ()* {{.*}}, bitcast (i32* @_end to void ()*)
; CHECK-FAST: or i1
; CHECK-FAST: br i1
; CHECK-FAST: call void ()* (void ()*)* @zzz(void ()*
; CHECK-FAST: br label
; CHECK-FAST: [[A:%[01-9a-z_.]+]] = phi i32 (i32, i32)* [ %f, %entry ], [ {{.*}} ]
; CHECK-FAST: call i32 {{.*}}[[A]](i32 {{.*}}, i32 {{.*}})
; CHECK-FAST: ret i32
