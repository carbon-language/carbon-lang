; RUN: opt -loop-reduce -S < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; LSR used not to be able to generate a float* induction variable in
; these cases due to scalar evolution not propagating nsw from an
; instruction to the SCEV, preventing distributing sext into the
; corresponding addrec.

define float @testadd(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @testadd
; CHECK: sext i32 %offset to i64
; CHECK: loop:
; CHECK-DAG: phi float*
; CHECK-DAG: phi i32
; CHECK-NOT: sext

entry:
  br label %loop

loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %sum = phi float [ %nextsum, %loop ], [ 0.000000e+00, %entry ]
  %index32 = add nuw nsw i32 %i, %offset
  %index64 = sext i32 %index32 to i64
  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %addend = load float, float* %ptr, align 4
  %nextsum = fadd float %sum, %addend
  %nexti = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret float %nextsum
}
