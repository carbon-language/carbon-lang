; RUN: opt < %s -dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i8 @load8(i8* %p) {
  ; CHECK: @"dfs$load8"
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: load
  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK: ret i8
  %a = load i8* %p
  ret i8 %a
}

define i16 @load16(i16* %p) {
  ; CHECK: @"dfs$load16"
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: load
  ; CHECK: load
  ; CHECK: icmp ne
  ; CHECK: call{{.*}}__dfsan_union
  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK: ret i16
  %a = load i16* %p
  ret i16 %a
}

define i32 @load32(i32* %p) {
  ; CHECK: @"dfs$load32"
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: bitcast
  ; CHECK: load
  ; CHECK: trunc
  ; CHECK: shl
  ; CHECK: lshr
  ; CHECK: or
  ; CHECK: icmp eq

  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK: ret i32

  ; CHECK: call{{.*}}__dfsan_union_load

  %a = load i32* %p
  ret i32 %a
}

define i64 @load64(i64* %p) {
  ; CHECK: @"dfs$load64"
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: bitcast
  ; CHECK: load
  ; CHECK: trunc
  ; CHECK: shl
  ; CHECK: lshr
  ; CHECK: or
  ; CHECK: icmp eq

  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK: ret i64

  ; CHECK: call{{.*}}__dfsan_union_load

  ; CHECK: getelementptr
  ; CHECK: load
  ; CHECK: icmp eq

  %a = load i64* %p
  ret i64 %a
}
