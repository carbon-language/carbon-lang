; RUN: opt -dse -S %s | FileCheck %s
; Note that we could do better by merging the two stores into one.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @test(i32* %P) {
  store i32 0, i32* %P
; CHECK: store i32
  %Q = bitcast i32* %P to i16*
  store i16 1, i16* %Q
; CHECK: store i16
  ret void
}
