; RUN: opt < %s -dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @store8(i8 %v, i8* %p) {
  ; CHECK: @"dfs$store8"
  ; CHECK: load{{.*}}__dfsan_arg_tls
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: store
  store i8 %v, i8* %p
  ret void
}

define void @store16(i16 %v, i16* %p) {
  ; CHECK: @"dfs$store16"
  ; CHECK: load{{.*}}__dfsan_arg_tls
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: store
  store i16 %v, i16* %p
  ret void
}

define void @store32(i32 %v, i32* %p) {
  ; CHECK: @"dfs$store32"
  ; CHECK: load{{.*}}__dfsan_arg_tls
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: store
  store i32 %v, i32* %p
  ret void
}

define void @store64(i64 %v, i64* %p) {
  ; CHECK: @"dfs$store64"
  ; CHECK: load{{.*}}__dfsan_arg_tls
  ; CHECK: ptrtoint
  ; CHECK: and
  ; CHECK: mul
  ; CHECK: inttoptr
  ; CHECK: insertelement
  ; CHECK: insertelement
  ; CHECK: insertelement
  ; CHECK: insertelement
  ; CHECK: insertelement
  ; CHECK: insertelement
  ; CHECK: insertelement
  ; CHECK: insertelement
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: store
  ; CHECK: store
  store i64 %v, i64* %p
  ret void
}
