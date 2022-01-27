; RUN: opt -licm -basic-aa < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

define void @f_0(i1 %p) nounwind ssp {
; CHECK-LABEL: @f_0(
entry:
  br label %for.body

for.body:
  br i1 undef, label %if.then, label %for.cond.backedge

for.cond.backedge:
  br i1 undef, label %for.end104, label %for.body

if.then:
  br i1 undef, label %if.then27, label %if.end.if.end.split_crit_edge.critedge

if.then27:
; CHECK: tail call void @llvm.assume
  tail call void @llvm.assume(i1 %p)
  br label %for.body61.us

if.end.if.end.split_crit_edge.critedge:
  br label %for.body61

for.body61.us:
  br i1 undef, label %for.cond.backedge, label %for.body61.us

for.body61:
  br i1 undef, label %for.cond.backedge, label %for.body61

for.end104:
  ret void
}

define void @f_1(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @f_1(
; CHECK-LABEL: entry:
; CHECK: call void @llvm.assume(i1 %cond)
; CHECK: %val = load i32, i32* %ptr
; CHECK-LABEL: loop:

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void @llvm.assume(i1 %cond)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

; Can't hoist because the call may throw and the assume
; may never execute.
define void @f_2(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @f_2(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK: call void @llvm.assume(i1 %cond)
; CHECK: %val = load i32, i32* %ptr

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void @maythrow()
  call void @llvm.assume(i1 %cond)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

; Note: resulting loop could be peeled and then hoisted, but
; by default assume is captured in phi cycle.
define void @f_3(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @f_3(
; CHECK-LABEL: entry:
; CHECK: %val = load i32, i32* %ptr
; CHECK-LABEL: loop:
; CHECK: call void @llvm.assume(i1 %x.cmp)

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  %x.cmp = phi i1 [%cond, %entry], [%cond.next, %loop]
  call void @llvm.assume(i1 %x.cmp)
  %val = load i32, i32* %ptr
  %cond.next = icmp eq i32 %val, 5
  %x.inc = add i32 %x, %val
  br label %loop
}


declare void @maythrow()
declare void @llvm.assume(i1)
