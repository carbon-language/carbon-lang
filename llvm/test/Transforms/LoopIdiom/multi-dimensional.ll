; RUN: opt -basicaa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%struct.ham = type { [2 x [2 x [2 x [16 x [8 x i32]]]]], i32, %struct.zot }
%struct.zot = type { i32, i16, i16, [2 x [1152 x i32]] }

define void @test1(%struct.ham* nocapture %arg) nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb11, %bb
  %tmp = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  br label %bb2

bb2:                                              ; preds = %bb2, %bb1
  %tmp3 = phi i64 [ 0, %bb1 ], [ %tmp8, %bb2 ]
  %tmp4 = getelementptr inbounds %struct.ham* %arg, i64 0, i32 0, i64 0, i64 1, i64 1, i64 %tmp, i64 %tmp3
  store i32 0, i32* %tmp4, align 4
  %tmp5 = getelementptr inbounds %struct.ham* %arg, i64 0, i32 0, i64 0, i64 1, i64 0, i64 %tmp, i64 %tmp3
  store i32 0, i32* %tmp5, align 4
  %tmp6 = getelementptr inbounds %struct.ham* %arg, i64 0, i32 0, i64 0, i64 0, i64 1, i64 %tmp, i64 %tmp3
  store i32 0, i32* %tmp6, align 4
  %tmp7 = getelementptr inbounds %struct.ham* %arg, i64 0, i32 0, i64 0, i64 0, i64 0, i64 %tmp, i64 %tmp3
  store i32 0, i32* %tmp7, align 4
  %tmp8 = add i64 %tmp3, 1
  %tmp9 = trunc i64 %tmp8 to i32
  %tmp10 = icmp eq i32 %tmp9, 8
  br i1 %tmp10, label %bb11, label %bb2

bb11:                                             ; preds = %bb2
  %tmp12 = add i64 %tmp, 1
  %tmp13 = trunc i64 %tmp12 to i32
  %tmp14 = icmp eq i32 %tmp13, 16
  br i1 %tmp14, label %bb15, label %bb1

bb15:                                             ; preds = %bb11
  ret void

; CHECK: @test1
; CHECK: bb1:
; CHECK-NOT: store
; CHECK: call void @llvm.memset.p0i8.i64
; CHECK-NEXT: call void @llvm.memset.p0i8.i64
; CHECK-NEXT: call void @llvm.memset.p0i8.i64
; CHECK-NEXT: call void @llvm.memset.p0i8.i64
; CHECK-NOT: store
; CHECK: br
}
