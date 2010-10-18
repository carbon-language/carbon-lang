; RUN: opt -tbaa -licm -enable-tbaa -S < %s | FileCheck %s

; LICM should be able to hoist the address load out of the loop
; by using TBAA information.

; CHECK:      entry:
; CHECK-NEXT:   %tmp3 = load double** @P, !tbaa !0
; CHECK-NEXT:   br label %for.body

@P = common global double* null

define void @foo(i64 %n) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %tmp3 = load double** @P, !tbaa !1
  %scevgep = getelementptr double* %tmp3, i64 %i.07
  %tmp4 = load double* %scevgep, !tbaa !2
  %mul = fmul double %tmp4, 2.300000e+00
  store double %mul, double* %scevgep, !tbaa !2
  %inc = add i64 %i.07, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

!0 = metadata !{metadata !"root", null}
!1 = metadata !{metadata !"pointer", metadata !0}
!2 = metadata !{metadata !"double", metadata !0}
