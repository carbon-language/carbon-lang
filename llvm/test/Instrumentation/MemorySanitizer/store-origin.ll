; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS1 %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=2 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS2 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Check origin instrumentation of stores

define void @Store(i32* nocapture %p, i32 %x) nounwind uwtable sanitize_memory {
entry:
  store i32 %x, i32* %p, align 4
  ret void
}

; CHECK: @Store
; CHECK: load {{.*}} @__msan_param_tls
; CHECK: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK: store
; CHECK: icmp
; CHECK: br i1
; CHECK: <label>

; Origin tracking level 1: simply store the origin value
; CHECK-ORIGINS1: store i32 {{.*}}[[ORIGIN]],

; Origin tracking level 2: pass origin value through __msan_chain_origin and store the result.
; CHECK-ORIGINS2: [[ORIGIN2:%[01-9a-z]+]] = call i32 @__msan_chain_origin(i32 {{.*}}[[ORIGIN]])
; CHECK-ORIGINS2: store i32 {{.*}}[[ORIGIN2]],

; CHECK: br label
; CHECK: <label>
; CHECK: store
; CHECK: ret void
