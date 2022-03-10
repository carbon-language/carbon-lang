; RUN: opt -mtriple=amdgcn-- -O3 -S %s | FileCheck %s

; Check that loop unswitch does not happen if condition is divergent.

; CHECK-LABEL: {{^}}define amdgpu_kernel void @divergent_unswitch
; CHECK: entry:
; CHECK: icmp
; CHECK: [[IF_COND:%[a-z0-9]+]] = icmp {{.*}} 567890
; CHECK: br label
; CHECK: br i1 [[IF_COND]]

define amdgpu_kernel void @divergent_unswitch(i32 * nocapture %out, i32 %n) {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %call = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %cmp2 = icmp eq i32 %call, 567890
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.inc
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.010 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  br i1 %cmp2, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32 * %out, i32 %i.010
  store i32 %i.010, i32 * %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i32 %i.010, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
