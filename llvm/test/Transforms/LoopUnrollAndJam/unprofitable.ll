; RUN: opt -loop-unroll-and-jam -pass-remarks=loop-unroll < %s -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.main-arm-none-eabi"

;; Common check for all tests. None should be unroll and jammed due to profitability
; CHECK-NOT: remark: {{.*}} unroll and jammed


; CHECK-LABEL: unprof1
; Multiple inner loop blocks
define void @unprof1(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner2 ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner2 ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner2 ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4, !tbaa !5
  %add.us = add i32 %0, %sum1.us
br label %for.inner2

for.inner2:
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner2 ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4, !tbaa !5
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unprof2
; Constant inner loop count
define void @unprof2(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4, !tbaa !5
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, 10
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4, !tbaa !5
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unprof3
; Complex inner loop
define void @unprof3(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4, !tbaa !5
  %add.us = add i32 %0, %sum1.us
  %add.us0 = add i32 %0, %sum1.us
  %add.us1 = add i32 %0, %sum1.us
  %add.us2 = add i32 %0, %sum1.us
  %add.us3 = add i32 %0, %sum1.us
  %add.us4 = add i32 %0, %sum1.us
  %add.us5 = add i32 %0, %sum1.us
  %add.us6 = add i32 %0, %sum1.us
  %add.us7 = add i32 %0, %sum1.us
  %add.us8 = add i32 %0, %sum1.us
  %add.us9 = add i32 %0, %sum1.us
  %add.us10 = add i32 %0, %sum1.us
  %add.us11 = add i32 %0, %sum1.us
  %add.us12 = add i32 %0, %sum1.us
  %add.us13 = add i32 %0, %sum1.us
  %add.us14 = add i32 %0, %sum1.us
  %add.us15 = add i32 %0, %sum1.us
  %add.us16 = add i32 %0, %sum1.us
  %add.us17 = add i32 %0, %sum1.us
  %add.us18 = add i32 %0, %sum1.us
  %add.us19 = add i32 %0, %sum1.us
  %add.us20 = add i32 %0, %sum1.us
  %add.us21 = add i32 %0, %sum1.us
  %add.us22 = add i32 %0, %sum1.us
  %add.us23 = add i32 %0, %sum1.us
  %add.us24 = add i32 %0, %sum1.us
  %add.us25 = add i32 %0, %sum1.us
  %add.us26 = add i32 %0, %sum1.us
  %add.us27 = add i32 %0, %sum1.us
  %add.us28 = add i32 %0, %sum1.us
  %add.us29 = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4, !tbaa !5
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; CHECK-LABEL: unprof4
; No loop invariant loads
define void @unprof4(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
; CHECK: %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
; CHECK: %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %j2 = add i32 %j.us, %i.us
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j2
  %0 = load i32, i32* %arrayidx.us, align 4, !tbaa !5
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4, !tbaa !5
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


attributes #0 = { "target-cpu"="cortex-m33" }

!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
