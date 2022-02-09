; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-delicm -analyze < %s | FileCheck %s -match-full-lines
;
; Load (but not store) of A[j] hoisted, reduction not written in all iterations.
; FIXME: %join is not mapped because the MemoryKind::Value mapping does not
;        encompass all iterations (last iteration is unused)
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) { /* reduction */
;          if (i != 2) {
;            phi += 4.2;
;            A[j] = phi;
;          }
;        }
;      }
;    }
;
define void @func(double* noalias nonnull %A) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %reduction.preheader, label %outer.exit


    reduction.preheader:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      %init = load double, double* %A_idx
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [%init, %reduction.preheader], [%join, %reduction.inc]
      br label %body



        body:
          %cond = icmp ne i32 %i, 2
          br i1 %cond, label %body_true, label %body_join

        body_true:
          %add = fadd double %phi, 4.2
          store double %add, double* %A_idx
          br label %body_join

        body_join:
          %join = phi double [%phi, %body], [%add, %body_true]
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 4
      br i1 %i.cmp, label %reduction.for, label %reduction.exit

    reduction.exit:
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; CHECK: Statistics {
; CHECK:     Compatible overwrites: 1
; CHECK:     Overwrites mapped to:  1
; CHECK:     Value scalars mapped:  1
; CHECK:     PHI scalars mapped:    1
; CHECK: }

; CHECK-NEXT: After accesses {
; CHECK-NEXT:     Stmt_reduction_preheader
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_reduction_preheader[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_preheader[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_preheader[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_reduction_for
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_join__phi[] };
; CHECK-NEXT:     Stmt_body_true
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body_true[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_body_true[i0, i1] -> MemRef_A[i0] : i1 <= 1; Stmt_body_true[i0, 3] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_body_true[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body_true[i0, i1] -> MemRef_join__phi[] };
; CHECK-NEXT:     Stmt_body_join
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body_join[i0, i1] -> MemRef_join[] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body_join[i0, i1] -> MemRef_join__phi[] };
; CHECK-NEXT:     Stmt_reduction_inc
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_join[] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] : i1 <= 2 };
; CHECK-NEXT: }
