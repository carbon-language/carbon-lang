; REQUIRES: asserts
; RUN: llc < %s -march=thumb -mcpu=swift -pre-RA-sched=source -enable-misched -verify-misched -debug-only=misched -o - 2>&1 > /dev/null | FileCheck %s
;
; Loop counter copies should be eliminated.
; There is also a MUL here, but we don't care where it is scheduled.
; CHECK: postinc
; CHECK: *** Final schedule for BB#2 ***
; CHECK: t2LDRs
; CHECK: t2ADDrr
; CHECK: t2CMPrr
; CHECK: COPY
define i32 @postinc(i32 %a, i32* nocapture %d, i32 %s) nounwind {
entry:
  %cmp4 = icmp eq i32 %a, 0
  br i1 %cmp4, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %s.05 = phi i32 [ %mul, %for.body ], [ 0, %entry ]
  %indvars.iv.next = add i32 %indvars.iv, %s
  %arrayidx = getelementptr inbounds i32* %d, i32 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %mul = mul nsw i32 %0, %s.05
  %exitcond = icmp eq i32 %indvars.iv.next, %a
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %mul, %for.body ]
  ret i32 %s.0.lcssa
}
