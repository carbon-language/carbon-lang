; RUN: opt -S -slp-vectorizer %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

define i64 @mismatched_intrinsics(<4 x i32> %in1, <2 x i32> %in2) nounwind {
; CHECK-LABEL: @mismatched_intrinsics
; CHECK: call i64 @llvm.arm64.neon.saddlv.i64.v4i32
; CHECK: call i64 @llvm.arm64.neon.saddlv.i64.v2i32

  %vaddlvq_s32.i = tail call i64 @llvm.arm64.neon.saddlv.i64.v4i32(<4 x i32> %in1) #2
  %vaddlv_s32.i = tail call i64 @llvm.arm64.neon.saddlv.i64.v2i32(<2 x i32> %in2) #2
  %tst = icmp sgt i64 %vaddlvq_s32.i, %vaddlv_s32.i
  %equal = sext i1 %tst to i64
  ret i64 %equal
}

declare i64 @llvm.arm64.neon.saddlv.i64.v4i32(<4 x i32> %in1)
declare i64 @llvm.arm64.neon.saddlv.i64.v2i32(<2 x i32> %in1)
