; RUN: opt %loadPolly -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; Rematerialize a load in the presence of a non-store WRITE access.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
;
; bodyB:
;   A[j] = val;
;
; bodyC:
;   memset(A, 0, 16);
;   memset(B, 0, 16);
; }
;

declare void @llvm.memset.p0f64.i64(double* nocapture, i8, i64, i32, i1)

define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %B_idx = getelementptr inbounds double, double* %B, i32 %j
      %val = load double, double* %B_idx
      br label %bodyB

    bodyB:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      store double %val, double* %A_idx
      br label %bodyC

    bodyC:
      call void @llvm.memset.p0f64.i64(double* %A, i8 0, i64 16, i32 1, i1 false)
      call void @llvm.memset.p0f64.i64(double* %B, i8 0, i64 16, i32 1, i1 false)
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: Statistics {
; CHECK:     Known loads forwarded: 1
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }
