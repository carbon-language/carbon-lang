; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define void @no_split_on_non_byte_width(i32) {
; This tests that allocas are not split into slices that are not byte width multiple
  %arg = alloca i32 , align 8
  store i32 %0, i32* %arg
  br label %load_i32

load_i32:
; CHECK-LABEL: load_i32:
; CHECK-NOT: bitcast {{.*}} to i1
; CHECK-NOT: zext i1
  %r0 = load i32* %arg
  br label %load_i1

load_i1:
; CHECK-LABEL: load_i1:
; CHECK: bitcast {{.*}} to i1
  %p1 = bitcast i32* %arg to i1*
  %t1 = load i1* %p1
  ret void
}
