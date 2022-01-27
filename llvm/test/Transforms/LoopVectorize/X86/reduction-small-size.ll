; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -mcpu=core-axv2 -force-vector-interleave=1 -dce -instcombine -debug-only=loop-vectorize -S < %s 2>&1  | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Make sure we ignore the costs of the redundant reduction casts
; char reduction_i8(char *a, char *b, int n) {
;   char sum = 0;
;   for (int i = 0; i < n; ++i)
;     sum += (a[i] + b[i]);
;   return sum;
; }
;

; CHECK-LABEL: reduction_i8
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = phi
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = phi
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = getelementptr
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = load
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = zext i8 %{{.*}} to i32
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = getelementptr
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = load
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = zext i8 %{{.*}} to i32
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = and i32 %{{.*}}, 255
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = trunc
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = icmp
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   br
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = phi
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = phi
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = getelementptr
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = load
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = zext i8 %{{.*}} to i32
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = getelementptr
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = load
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = zext i8 %{{.*}} to i32
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = and i32 %{{.*}}, 255
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = trunc
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   %{{.*}} = icmp
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 2 For instruction:   br
;
define i8 @reduction_i8(i8* nocapture readonly %a, i8* nocapture readonly %b, i32 %n) {
entry:
  %cmp.12 = icmp sgt i32 %n, 0
  br i1 %cmp.12, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  %add5.lcssa = phi i32 [ %add5, %for.body ]
  %conv6 = trunc i32 %add5.lcssa to i8
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i8 [ %conv6, %for.cond.for.cond.cleanup_crit_edge ], [ 0, %entry ]
  ret i8 %sum.0.lcssa

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %sum.013 = phi i32 [ %add5, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds i8, i8* %b, i64 %indvars.iv
  %1 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %conv4 = and i32 %sum.013, 255
  %add = add nuw nsw i32 %conv, %conv4
  %add5 = add nuw nsw i32 %add, %conv3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}
