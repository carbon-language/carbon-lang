; RUN: llc < %s -march=x86-64 | FileCheck %s

; LSR's OptimizeMax should eliminate the select (max).

; CHECK: foo:
; CHECK-NOT: cmov
; CHECK: jle

define void @foo(i64 %n, double* nocapture %p) nounwind {
entry:
  %cmp6 = icmp slt i64 %n, 0                      ; <i1> [#uses=1]
  br i1 %cmp6, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %tmp = icmp sgt i64 %n, 0                       ; <i1> [#uses=1]
  %n.op = add i64 %n, 1                           ; <i64> [#uses=1]
  %tmp1 = select i1 %tmp, i64 %n.op, i64 1        ; <i64> [#uses=1]
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i = phi i64 [ %i.next, %for.body ], [ 0, %for.body.preheader ] ; <i64> [#uses=2]
  %arrayidx = getelementptr double* %p, i64 %i    ; <double*> [#uses=2]
  %t4 = load double* %arrayidx                    ; <double> [#uses=1]
  %mul = fmul double %t4, 2.200000e+00            ; <double> [#uses=1]
  store double %mul, double* %arrayidx
  %i.next = add nsw i64 %i, 1                     ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %i.next, %tmp1          ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
