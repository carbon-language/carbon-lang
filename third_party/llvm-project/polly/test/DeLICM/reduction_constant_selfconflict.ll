; RUN: opt %loadPolly -polly-flatten-schedule -polly-delicm-compute-known=true -polly-delicm -analyze < %s | FileCheck %s
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) { /* reduction */
;          phi += 4.2;
;          A[j] = phi;
;        }
;        A[j] = phi;
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
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %reduction.preheader], [%add, %reduction.inc]
      %i.cmp = icmp slt i32 %i, 4
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      br i1 %i.cmp, label %body, label %reduction.exit



        body:
          %add = fadd double %phi, 4.2
          store double %add, double* %A_idx
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
      store double %phi, double* %A_idx
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_reduction_preheader
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
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_reduction_inc
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:            new: {  Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] : i1 <= 7 - 5i0; Stmt_reduction_inc[1, 3] -> MemRef_A[1] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_reduction_exit
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
