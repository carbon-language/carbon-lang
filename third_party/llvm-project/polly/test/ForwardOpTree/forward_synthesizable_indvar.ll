; RUN: opt %loadPolly -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; Test support for (synthesizable) inducation variables.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = j;
;
; bodyB:
;   A[0] = val;
; }
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %val = sitofp i32 %j to double
      br label %bodyB

    bodyB:
      store double %val, double* %A
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
; CHECK:     Instructions copied: 1
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = sitofp i32 %j to double
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = sitofp i32 %j to double
; CHECK-NEXT:                   store double %val, double* %A, align 8
; CHECK-NEXT:             }
; CHECK-NEXT: }
