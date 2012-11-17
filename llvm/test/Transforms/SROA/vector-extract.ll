; RUN: opt < %s -sroa -S | FileCheck %s
; rdar://12713675

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define <2 x i16> @test1(i64 %x) nounwind ssp {
; CHECK: @test1
entry:
  %tmp = alloca i64, align 8
  br i1 undef, label %bb1, label %bb2
; CHECK-NOT: alloca

bb1:
  store i64 %x, i64* %tmp, align 8
; CHECK-NOT: store
  %0 = bitcast i64* %tmp to <2 x i16>*
  %1 = load <2 x i16>* %0, align 8
; CHECK-NOT: load
; CHECK: trunc i64 %x to i32
; CHECK: bitcast i32
  ret <2 x i16> %1

bb2:
  ret <2 x i16> < i16 0, i16 0 >
}
