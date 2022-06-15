; RUN: opt %loadPolly -polly-print-delicm -disable-output < %s | FileCheck %s
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        fsomeval = 21.0 + 21.0;
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) /* reduction */
;          phi += 4.2;
;        A[j] = fsomeval;
;      }
;      g(fsomeval);
;    }
;
; Check that fsomeval is not mapped to A[j] because it is escaping the SCoP.
; Supporting this would require reloading the scalar from A[j], and/or
; identifying the last instance of fsomeval that escapes.
;
define void @func(double* noalias nonnull %A) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  %fsomeval = fadd double 21.0, 21.0
  br i1 %j.cmp, label %reduction.preheader, label %outer.exit


    reduction.preheader:
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %reduction.preheader], [%add, %reduction.inc]
      %i.cmp = icmp slt i32 %i, 4
      br i1 %i.cmp, label %body, label %reduction.exit



        body:
          %add = fadd double %phi, 4.2
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      store double %fsomeval, double* %A_idx
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  call void @g(double %fsomeval)
  ret void
}

declare void @g(double)


; CHECK: Statistics {
; CHECK:     Compatible overwrites: 1
; CHECK: }
; CHECK: No modification has been made
