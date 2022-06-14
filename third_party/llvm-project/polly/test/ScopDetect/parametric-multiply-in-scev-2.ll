; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s


; CHECK-NOT: Valid Region
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @blam(float* %A, float* %B) {
bb:
  %tmp1 = alloca i64
  %tmp2 = shl i64 2, undef
  %tmp3 = shl i64 2, undef
  %tmp4 = mul nsw i64 %tmp2, %tmp3
  br label %loop

loop:
  %indvar = phi i64 [ %indvar.next, %loop ], [ 0, %bb ]
  %gep = getelementptr inbounds i64, i64* %tmp1, i64 %indvar
  %tmp12 = load i64, i64* %gep
  %tmp13 = mul nsw i64 %tmp12, %tmp4
  %ptr = getelementptr inbounds float, float* %B, i64 %tmp13
  %val = load float, float* %ptr
  store float %val, float* %A
  %indvar.next = add nsw i64 %indvar, 1
  br i1 false, label %loop, label %bb21

bb21:
  ret void
}
