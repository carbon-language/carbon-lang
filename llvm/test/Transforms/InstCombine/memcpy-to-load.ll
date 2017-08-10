; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

; Make sure that the memcpy has been replaced with a load/store of i64.

define void @foo(i8* %d, i8* %s) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i8* %s to i64*
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* %d to i64*
; CHECK-NEXT:    [[TMP3:%.*]] = load i64, i64* [[TMP1]], align 1
; CHECK-NEXT:    store i64 [[TMP3]], i64* [[TMP2]], align 1
; CHECK-NEXT:    ret void
;
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %d, i8* %s, i32 8, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
