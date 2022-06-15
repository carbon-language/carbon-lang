; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-invariant-load-hoisting -polly-flatten-schedule -polly-delicm-overapproximate-writes=true -polly-delicm-compute-known=true -polly-print-delicm -disable-output < %s | FileCheck %s
;
;    void func(int *A, int* StartPtr) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        int Start = *Startptr;
;        int phi = Start;
;        for (int i = Start; i < 4; i += 1) /* reduction */
;          phi += 42;
;        A[j] = phi;
;      }
;    }
;
define void @func(i32* noalias nonnull %A, i32* noalias nonnull %StartPtr) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %reduction.preheader, label %outer.exit


    reduction.preheader:
      %Start = load i32, i32* %StartPtr
      br label %reduction.for

    reduction.for:
      %i = phi i32 [%Start, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi i32 [%Start, %reduction.preheader], [%mul, %reduction.inc]
      br label %body



        body:
          %mul = mul i32 %phi, 2
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 4
      br i1 %i.cmp, label %reduction.for, label %reduction.exit

    reduction.exit:
      %A_idx = getelementptr inbounds i32, i32* %A, i32 %j
      store i32 %mul, i32* %A_idx
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}

; FIXME: The accesses for %i should not be there because, with
; load-hoisting %Start is an affine loop. To be fixed in ScopBuilder.

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_reduction_preheader
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_reduction_preheader[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_reduction_preheader[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_reduction_for
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_reduction_for[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] : -i0 < i1 <= 3 - Start; Stmt_reduction_for[1, 0] -> MemRef_A[1] : Start >= 4; Stmt_reduction_for[0, 0] -> MemRef_A[0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_reduction_for[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] : i0 <= i1 <= 3 - Start; Stmt_reduction_for[i0, 0] -> MemRef_A[i0] : i0 <= -4 + Start; Stmt_reduction_for[1, 0] -> MemRef_A[1] : Start <= 4 }
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_body[i0, i1] -> MemRef_mul[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_body[i0, i1] -> MemRef_A[i0] : i0 <= i1 <= 3 - Start; Stmt_body[i0, 0] -> MemRef_A[i0] : i0 <= -4 + Start; Stmt_body[1, 0] -> MemRef_A[1] : Start <= 4 }
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_body[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_body[i0, i1] -> MemRef_A[i0] : i0 <= i1 <= 3 - Start; Stmt_body[1, 0] -> MemRef_A[1]; Stmt_body[0, 0] -> MemRef_A[0] : Start >= 4 };
; CHECK-NEXT:     Stmt_reduction_inc
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_reduction_inc[i0, i1] -> MemRef_mul[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] : i0 <= i1 <= 3 - Start; Stmt_reduction_inc[1, 0] -> MemRef_A[1]; Stmt_reduction_inc[0, 0] -> MemRef_A[0] : Start >= 4 };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] : 0 < i1 <= 3 - Start; Stmt_reduction_inc[i0, 0] -> MemRef_A[i0] }
; CHECK-NEXT:     Stmt_reduction_exit
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [Start] -> { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [Start] -> { Stmt_reduction_exit[i0] -> MemRef_mul[] };
; CHECK-NEXT:            new: [Start] -> { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
