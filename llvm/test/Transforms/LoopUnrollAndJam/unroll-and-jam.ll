; RUN: opt -basicaa -tbaa -loop-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.main-arm-none-eabi"

; CHECK-LABEL: test1
; Tests for(i) { sum = 0; for(j) sum += B[j]; A[i] = sum; }
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i32 [[J:%.*]], 0
; CHECK-NEXT:    [[CMP122:%.*]] = icmp ne i32 [[I:%.*]], 0
; CHECK-NEXT:    [[OR_COND:%.*]] = and i1 [[CMP]], [[CMP122]]
; CHECK-NEXT:    br i1 [[OR_COND]], label [[FOR_OUTER_PREHEADER:%.*]], label [[FOR_END:%.*]]
; CHECK:       for.outer.preheader:
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[I]], -1
; CHECK-NEXT:    [[XTRAITER:%.*]] = and i32 [[I]], 3
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[TMP0]], 3
; CHECK-NEXT:    br i1 [[TMP1]], label [[FOR_END_LOOPEXIT_UNR_LCSSA:%.*]], label [[FOR_OUTER_PREHEADER_NEW:%.*]]
; CHECK:       for.outer.preheader.new:
; CHECK-NEXT:    [[UNROLL_ITER:%.*]] = sub i32 [[I]], [[XTRAITER]]
; CHECK-NEXT:    br label [[FOR_OUTER:%.*]]
; CHECK:       for.outer:
; CHECK-NEXT:    [[I_US:%.*]] = phi i32 [ [[ADD8_US_3:%.*]], [[FOR_LATCH:%.*]] ], [ 0, [[FOR_OUTER_PREHEADER_NEW]] ]
; CHECK-NEXT:    [[NITER:%.*]] = phi i32 [ [[UNROLL_ITER]], [[FOR_OUTER_PREHEADER_NEW]] ], [ [[NITER_NSUB_3:%.*]], [[FOR_LATCH]] ]
; CHECK-NEXT:    [[ADD8_US:%.*]] = add nuw nsw i32 [[I_US]], 1
; CHECK-NEXT:    [[NITER_NSUB:%.*]] = sub i32 [[NITER]], 1
; CHECK-NEXT:    [[ADD8_US_1:%.*]] = add nuw nsw i32 [[ADD8_US]], 1
; CHECK-NEXT:    [[NITER_NSUB_1:%.*]] = sub i32 [[NITER_NSUB]], 1
; CHECK-NEXT:    [[ADD8_US_2:%.*]] = add nuw nsw i32 [[ADD8_US_1]], 1
; CHECK-NEXT:    [[NITER_NSUB_2:%.*]] = sub i32 [[NITER_NSUB_1]], 1
; CHECK-NEXT:    [[ADD8_US_3]] = add nuw i32 [[ADD8_US_2]], 1
; CHECK-NEXT:    [[NITER_NSUB_3]] = sub i32 [[NITER_NSUB_2]], 1
; CHECK-NEXT:    br label [[FOR_INNER:%.*]]
; CHECK:       for.inner:
; CHECK-NEXT:    [[J_US:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[INC_US:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[SUM1_US:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[ADD_US:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[J_US_1:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[INC_US_1:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[SUM1_US_1:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[ADD_US_1:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[J_US_2:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[INC_US_2:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[SUM1_US_2:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[ADD_US_2:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[J_US_3:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[INC_US_3:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[SUM1_US_3:%.*]] = phi i32 [ 0, [[FOR_OUTER]] ], [ [[ADD_US_3:%.*]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[ARRAYIDX_US:%.*]] = getelementptr inbounds i32, i32* [[B:%.*]], i32 [[J_US]]
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[ARRAYIDX_US]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD_US]] = add i32 [[TMP2]], [[SUM1_US]]
; CHECK-NEXT:    [[INC_US]] = add nuw i32 [[J_US]], 1
; CHECK-NEXT:    [[ARRAYIDX_US_1:%.*]] = getelementptr inbounds i32, i32* [[B]], i32 [[J_US_1]]
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[ARRAYIDX_US_1]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD_US_1]] = add i32 [[TMP3]], [[SUM1_US_1]]
; CHECK-NEXT:    [[INC_US_1]] = add nuw i32 [[J_US_1]], 1
; CHECK-NEXT:    [[ARRAYIDX_US_2:%.*]] = getelementptr inbounds i32, i32* [[B]], i32 [[J_US_2]]
; CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* [[ARRAYIDX_US_2]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD_US_2]] = add i32 [[TMP4]], [[SUM1_US_2]]
; CHECK-NEXT:    [[INC_US_2]] = add nuw i32 [[J_US_2]], 1
; CHECK-NEXT:    [[ARRAYIDX_US_3:%.*]] = getelementptr inbounds i32, i32* [[B]], i32 [[J_US_3]]
; CHECK-NEXT:    [[TMP5:%.*]] = load i32, i32* [[ARRAYIDX_US_3]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD_US_3]] = add i32 [[TMP5]], [[SUM1_US_3]]
; CHECK-NEXT:    [[INC_US_3]] = add nuw i32 [[J_US_3]], 1
; CHECK-NEXT:    [[EXITCOND_3:%.*]] = icmp eq i32 [[INC_US_3]], [[J]]
; CHECK-NEXT:    br i1 [[EXITCOND_3]], label [[FOR_LATCH]], label [[FOR_INNER]]
; CHECK:       for.latch:
; CHECK-NEXT:    [[ADD_US_LCSSA:%.*]] = phi i32 [ [[ADD_US]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[ADD_US_LCSSA_1:%.*]] = phi i32 [ [[ADD_US_1]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[ADD_US_LCSSA_2:%.*]] = phi i32 [ [[ADD_US_2]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[ADD_US_LCSSA_3:%.*]] = phi i32 [ [[ADD_US_3]], [[FOR_INNER]] ]
; CHECK-NEXT:    [[ARRAYIDX6_US:%.*]] = getelementptr inbounds i32, i32* [[A:%.*]], i32 [[I_US]]
; CHECK-NEXT:    store i32 [[ADD_US_LCSSA]], i32* [[ARRAYIDX6_US]], align 4, !tbaa !0
; CHECK-NEXT:    [[ARRAYIDX6_US_1:%.*]] = getelementptr inbounds i32, i32* [[A]], i32 [[ADD8_US]]
; CHECK-NEXT:    store i32 [[ADD_US_LCSSA_1]], i32* [[ARRAYIDX6_US_1]], align 4, !tbaa !0
; CHECK-NEXT:    [[ARRAYIDX6_US_2:%.*]] = getelementptr inbounds i32, i32* [[A]], i32 [[ADD8_US_1]]
; CHECK-NEXT:    store i32 [[ADD_US_LCSSA_2]], i32* [[ARRAYIDX6_US_2]], align 4, !tbaa !0
; CHECK-NEXT:    [[ARRAYIDX6_US_3:%.*]] = getelementptr inbounds i32, i32* [[A]], i32 [[ADD8_US_2]]
; CHECK-NEXT:    store i32 [[ADD_US_LCSSA_3]], i32* [[ARRAYIDX6_US_3]], align 4, !tbaa !0
; CHECK-NEXT:    [[NITER_NCMP_3:%.*]] = icmp eq i32 [[NITER_NSUB_3]], 0
; CHECK-NEXT:    br i1 [[NITER_NCMP_3]], label [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT:%.*]], label [[FOR_OUTER]], !llvm.loop !4
; CHECK:       for.end.loopexit.unr-lcssa.loopexit:
; CHECK-NEXT:    [[I_US_UNR_PH:%.*]] = phi i32 [ [[ADD8_US_3]], [[FOR_LATCH]] ]
; CHECK-NEXT:    br label [[FOR_END_LOOPEXIT_UNR_LCSSA]]
; CHECK:       for.end.loopexit.unr-lcssa:
; CHECK-NEXT:    [[I_US_UNR:%.*]] = phi i32 [ 0, [[FOR_OUTER_PREHEADER]] ], [ [[I_US_UNR_PH]], [[FOR_END_LOOPEXIT_UNR_LCSSA_LOOPEXIT]] ]
; CHECK-NEXT:    [[LCMP_MOD:%.*]] = icmp ne i32 [[XTRAITER]], 0
; CHECK-NEXT:    br i1 [[LCMP_MOD]], label [[FOR_OUTER_EPIL_PREHEADER:%.*]], label [[FOR_END_LOOPEXIT:%.*]]
; CHECK:       for.outer.epil.preheader:
; CHECK-NEXT:    br label [[FOR_OUTER_EPIL:%.*]]
; CHECK:       for.outer.epil:
; CHECK-NEXT:    br label [[FOR_INNER_EPIL:%.*]]
; CHECK:       for.inner.epil:
; CHECK-NEXT:    [[J_US_EPIL:%.*]] = phi i32 [ 0, [[FOR_OUTER_EPIL]] ], [ [[INC_US_EPIL:%.*]], [[FOR_INNER_EPIL]] ]
; CHECK-NEXT:    [[SUM1_US_EPIL:%.*]] = phi i32 [ 0, [[FOR_OUTER_EPIL]] ], [ [[ADD_US_EPIL:%.*]], [[FOR_INNER_EPIL]] ]
; CHECK-NEXT:    [[ARRAYIDX_US_EPIL:%.*]] = getelementptr inbounds i32, i32* [[B]], i32 [[J_US_EPIL]]
; CHECK-NEXT:    [[TMP6:%.*]] = load i32, i32* [[ARRAYIDX_US_EPIL]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD_US_EPIL]] = add i32 [[TMP6]], [[SUM1_US_EPIL]]
; CHECK-NEXT:    [[INC_US_EPIL]] = add nuw i32 [[J_US_EPIL]], 1
; CHECK-NEXT:    [[EXITCOND_EPIL:%.*]] = icmp eq i32 [[INC_US_EPIL]], [[J]]
; CHECK-NEXT:    br i1 [[EXITCOND_EPIL]], label [[FOR_LATCH_EPIL:%.*]], label [[FOR_INNER_EPIL]]
; CHECK:       for.latch.epil:
; CHECK-NEXT:    [[ADD_US_LCSSA_EPIL:%.*]] = phi i32 [ [[ADD_US_EPIL]], [[FOR_INNER_EPIL]] ]
; CHECK-NEXT:    [[ARRAYIDX6_US_EPIL:%.*]] = getelementptr inbounds i32, i32* [[A]], i32 [[I_US_UNR]]
; CHECK-NEXT:    store i32 [[ADD_US_LCSSA_EPIL]], i32* [[ARRAYIDX6_US_EPIL]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD8_US_EPIL:%.*]] = add nuw i32 [[I_US_UNR]], 1
; CHECK-NEXT:    [[EPIL_ITER_SUB:%.*]] = sub i32 [[XTRAITER]], 1
; CHECK-NEXT:    [[EPIL_ITER_CMP:%.*]] = icmp ne i32 [[EPIL_ITER_SUB]], 0
; CHECK-NEXT:    br i1 [[EPIL_ITER_CMP]], label [[FOR_OUTER_EPIL_1:%.*]], label [[FOR_END_LOOPEXIT_EPILOG_LCSSA:%.*]]
; CHECK:       for.end.loopexit.epilog-lcssa:
; CHECK-NEXT:    br label [[FOR_END_LOOPEXIT]]
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label [[FOR_END]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
; CHECK:       for.outer.epil.1:
; CHECK-NEXT:    br label [[FOR_INNER_EPIL_1:%.*]]
; CHECK:       for.inner.epil.1:
; CHECK-NEXT:    [[J_US_EPIL_1:%.*]] = phi i32 [ 0, [[FOR_OUTER_EPIL_1]] ], [ [[INC_US_EPIL_1:%.*]], [[FOR_INNER_EPIL_1]] ]
; CHECK-NEXT:    [[SUM1_US_EPIL_1:%.*]] = phi i32 [ 0, [[FOR_OUTER_EPIL_1]] ], [ [[ADD_US_EPIL_1:%.*]], [[FOR_INNER_EPIL_1]] ]
; CHECK-NEXT:    [[ARRAYIDX_US_EPIL_1:%.*]] = getelementptr inbounds i32, i32* [[B]], i32 [[J_US_EPIL_1]]
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[ARRAYIDX_US_EPIL_1]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD_US_EPIL_1]] = add i32 [[TMP7]], [[SUM1_US_EPIL_1]]
; CHECK-NEXT:    [[INC_US_EPIL_1]] = add nuw i32 [[J_US_EPIL_1]], 1
; CHECK-NEXT:    [[EXITCOND_EPIL_1:%.*]] = icmp eq i32 [[INC_US_EPIL_1]], [[J]]
; CHECK-NEXT:    br i1 [[EXITCOND_EPIL_1]], label [[FOR_LATCH_EPIL_1:%.*]], label [[FOR_INNER_EPIL_1]]
; CHECK:       for.latch.epil.1:
; CHECK-NEXT:    [[ADD_US_LCSSA_EPIL_1:%.*]] = phi i32 [ [[ADD_US_EPIL_1]], [[FOR_INNER_EPIL_1]] ]
; CHECK-NEXT:    [[ARRAYIDX6_US_EPIL_1:%.*]] = getelementptr inbounds i32, i32* [[A]], i32 [[ADD8_US_EPIL]]
; CHECK-NEXT:    store i32 [[ADD_US_LCSSA_EPIL_1]], i32* [[ARRAYIDX6_US_EPIL_1]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD8_US_EPIL_1:%.*]] = add nuw i32 [[ADD8_US_EPIL]], 1
; CHECK-NEXT:    [[EPIL_ITER_SUB_1:%.*]] = sub i32 [[EPIL_ITER_SUB]], 1
; CHECK-NEXT:    [[EPIL_ITER_CMP_1:%.*]] = icmp ne i32 [[EPIL_ITER_SUB_1]], 0
; CHECK-NEXT:    br i1 [[EPIL_ITER_CMP_1]], label [[FOR_OUTER_EPIL_2:%.*]], label [[FOR_END_LOOPEXIT_EPILOG_LCSSA]]
; CHECK:       for.outer.epil.2:
; CHECK-NEXT:    br label [[FOR_INNER_EPIL_2:%.*]]
; CHECK:       for.inner.epil.2:
; CHECK-NEXT:    [[J_US_EPIL_2:%.*]] = phi i32 [ 0, [[FOR_OUTER_EPIL_2]] ], [ [[INC_US_EPIL_2:%.*]], [[FOR_INNER_EPIL_2]] ]
; CHECK-NEXT:    [[SUM1_US_EPIL_2:%.*]] = phi i32 [ 0, [[FOR_OUTER_EPIL_2]] ], [ [[ADD_US_EPIL_2:%.*]], [[FOR_INNER_EPIL_2]] ]
; CHECK-NEXT:    [[ARRAYIDX_US_EPIL_2:%.*]] = getelementptr inbounds i32, i32* [[B]], i32 [[J_US_EPIL_2]]
; CHECK-NEXT:    [[TMP8:%.*]] = load i32, i32* [[ARRAYIDX_US_EPIL_2]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD_US_EPIL_2]] = add i32 [[TMP8]], [[SUM1_US_EPIL_2]]
; CHECK-NEXT:    [[INC_US_EPIL_2]] = add nuw i32 [[J_US_EPIL_2]], 1
; CHECK-NEXT:    [[EXITCOND_EPIL_2:%.*]] = icmp eq i32 [[INC_US_EPIL_2]], [[J]]
; CHECK-NEXT:    br i1 [[EXITCOND_EPIL_2]], label [[FOR_LATCH_EPIL_2:%.*]], label [[FOR_INNER_EPIL_2]]
; CHECK:       for.latch.epil.2:
; CHECK-NEXT:    [[ADD_US_LCSSA_EPIL_2:%.*]] = phi i32 [ [[ADD_US_EPIL_2]], [[FOR_INNER_EPIL_2]] ]
; CHECK-NEXT:    [[ARRAYIDX6_US_EPIL_2:%.*]] = getelementptr inbounds i32, i32* [[A]], i32 [[ADD8_US_EPIL_1]]
; CHECK-NEXT:    store i32 [[ADD_US_LCSSA_EPIL_2]], i32* [[ARRAYIDX6_US_EPIL_2]], align 4, !tbaa !0
; CHECK-NEXT:    [[ADD8_US_EPIL_2:%.*]] = add nuw i32 [[ADD8_US_EPIL_1]], 1
; CHECK-NEXT:    [[EPIL_ITER_SUB_2:%.*]] = sub i32 [[EPIL_ITER_SUB_1]], 1
; CHECK-NEXT:    br label [[FOR_END_LOOPEXIT_EPILOG_LCSSA]]
define void @test1(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
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

; CHECK-LABEL: test2
; Tests for(i) { sum = A[i]; for(j) sum += B[j]; A[i] = sum; }
; A[i] load/store dependency should not block unroll-and-jam
; CHECK: for.outer:
; CHECK:   %i.us = phi i32 [ %add9.us.3, %for.latch ], [ 0, %for.outer.preheader.new ]
; CHECK:   %niter = phi i32 [ %unroll_iter, %for.outer.preheader.new ], [ %niter.nsub.3, %for.latch ]
; CHECK:   br label %for.inner
; CHECK: for.inner:
; CHECK:   %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
; CHECK:   %sum1.us = phi i32 [ %2, %for.outer ], [ %add.us, %for.inner ]
; CHECK:   %j.us.1 = phi i32 [ 0, %for.outer ], [ %inc.us.1, %for.inner ]
; CHECK:   %sum1.us.1 = phi i32 [ %3, %for.outer ], [ %add.us.1, %for.inner ]
; CHECK:   %j.us.2 = phi i32 [ 0, %for.outer ], [ %inc.us.2, %for.inner ]
; CHECK:   %sum1.us.2 = phi i32 [ %4, %for.outer ], [ %add.us.2, %for.inner ]
; CHECK:   %j.us.3 = phi i32 [ 0, %for.outer ], [ %inc.us.3, %for.inner ]
; CHECK:   %sum1.us.3 = phi i32 [ %5, %for.outer ], [ %add.us.3, %for.inner ]
; CHECK:   br i1 %exitcond.3, label %for.latch, label %for.inner
; CHECK: for.latch:
; CHECK:   %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
; CHECK:   %add.us.lcssa.1 = phi i32 [ %add.us.1, %for.inner ]
; CHECK:   %add.us.lcssa.2 = phi i32 [ %add.us.2, %for.inner ]
; CHECK:   %add.us.lcssa.3 = phi i32 [ %add.us.3, %for.inner ]
; CHECK:   br i1 %niter.ncmp.3, label %for.end10.loopexit.unr-lcssa.loopexit, label %for.outer
; CHECK: for.end10.loopexit.unr-lcssa.loopexit:
define void @test2(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp125 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp125
  br i1 %or.cond, label %for.outer.preheader, label %for.end10

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add9.us, %for.latch ], [ 0, %for.outer.preheader ]
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  %0 = load i32, i32* %arrayidx.us, align 4, !tbaa !5
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ %0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %1 = load i32, i32* %arrayidx6.us, align 4, !tbaa !5
  %add.us = add i32 %1, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  store i32 %add.us.lcssa, i32* %arrayidx.us, align 4, !tbaa !5
  %add9.us = add nuw i32 %i.us, 1
  %exitcond28 = icmp eq i32 %add9.us, %I
  br i1 %exitcond28, label %for.end10.loopexit, label %for.outer

for.end10.loopexit:
  br label %for.end10

for.end10:
  ret void
}


; CHECK-LABEL: test3
; Tests Complete unroll-and-jam of the outer loop
; CHECK: for.outer:
; CHECK:   br label %for.inner
; CHECK: for.inner:
; CHECK:   %j.021 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
; CHECK:   %sum1.020 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
; CHECK:   %j.021.1 = phi i32 [ 0, %for.outer ], [ %inc.1, %for.inner ]
; CHECK:   %sum1.020.1 = phi i32 [ 0, %for.outer ], [ %add.1, %for.inner ]
; CHECK:   %j.021.2 = phi i32 [ 0, %for.outer ], [ %inc.2, %for.inner ]
; CHECK:   %sum1.020.2 = phi i32 [ 0, %for.outer ], [ %add.2, %for.inner ]
; CHECK:   %j.021.3 = phi i32 [ 0, %for.outer ], [ %inc.3, %for.inner ]
; CHECK:   %sum1.020.3 = phi i32 [ 0, %for.outer ], [ %add.3, %for.inner ]
; CHECK:   br i1 %exitcond.3, label %for.latch, label %for.inner
; CHECK: for.latch:
; CHECK:   %add.lcssa = phi i32 [ %add, %for.inner ]
; CHECK:   %add.lcssa.1 = phi i32 [ %add.1, %for.inner ]
; CHECK:   %add.lcssa.2 = phi i32 [ %add.2, %for.inner ]
; CHECK:   %add.lcssa.3 = phi i32 [ %add.3, %for.inner ]
; CHECK:   br label %for.end
; CHECK: for.end:
define void @test3(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
entry:
  %cmp = icmp eq i32 %J, 0
  br i1 %cmp, label %for.end, label %for.preheader

for.preheader:
  br label %for.outer

for.outer:
  %i.022 = phi i32 [ %add8, %for.latch ], [ 0, %for.preheader ]
  br label %for.inner

for.inner:
  %j.021 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1.020 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j.021
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !5
  %sub = add i32 %sum1.020, 10
  %add = sub i32 %sub, %0
  %inc = add nuw i32 %j.021, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i.022
  store i32 %add, i32* %arrayidx6, align 4, !tbaa !5
  %add8 = add nuw nsw i32 %i.022, 1
  %exitcond23 = icmp eq i32 %add8, 4
  br i1 %exitcond23, label %for.end, label %for.outer

for.end:
  ret void
}

; CHECK-LABEL: test4
; Tests Complete unroll-and-jam with a trip count of 1
; CHECK: for.outer:
; CHECK:   br label %for.inner
; CHECK: for.inner:
; CHECK:   %j.021 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
; CHECK:   %sum1.020 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
; CHECK:   br i1 %exitcond, label %for.latch, label %for.inner
; CHECK: for.latch:
; CHECK:   %add.lcssa = phi i32 [ %add, %for.inner ]
; CHECK:   br label %for.end
; CHECK: for.end:
define void @test4(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
entry:
  %cmp = icmp eq i32 %J, 0
  br i1 %cmp, label %for.end, label %for.preheader

for.preheader:
  br label %for.outer

for.outer:
  %i.022 = phi i32 [ %add8, %for.latch ], [ 0, %for.preheader ]
  br label %for.inner

for.inner:
  %j.021 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum1.020 = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j.021
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !5
  %sub = add i32 %sum1.020, 10
  %add = sub i32 %sub, %0
  %inc = add nuw i32 %j.021, 1
  %exitcond = icmp eq i32 %inc, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %i.022
  store i32 %add, i32* %arrayidx6, align 4, !tbaa !5
  %add8 = add nuw nsw i32 %i.022, 1
  %exitcond23 = icmp eq i32 %add8, 1
  br i1 %exitcond23, label %for.end, label %for.outer

for.end:
  ret void
}





; CHECK-LABEL: test5
; Multiple SubLoopBlocks
; CHECK: for.outer:
; CHECK:   br label %for.inner
; CHECK: for.inner:
; CHECK:   %inc8.sink15 = phi i32 [ 0, %for.outer ], [ %inc8, %for.inc.1 ]
; CHECK:   %inc8.sink15.1 = phi i32 [ 0, %for.outer ], [ %inc8.1, %for.inc.1 ]
; CHECK:   br label %for.inner2
; CHECK: for.inner2:
; CHECK:   br i1 %tobool, label %for.cond4, label %for.inc
; CHECK: for.cond4:
; CHECK:   br i1 %tobool.1, label %for.cond4a, label %for.inc
; CHECK: for.cond4a:
; CHECK:   br label %for.inc
; CHECK: for.inc:
; CHECK:   br i1 %tobool.11, label %for.cond4.1, label %for.inc.1
; CHECK: for.latch:
; CHECK:   br label %for.end
; CHECK: for.end:
; CHECK:   ret i32 0
; CHECK: for.cond4.1:
; CHECK:   br i1 %tobool.1.1, label %for.cond4a.1, label %for.inc.1
; CHECK: for.cond4a.1:
; CHECK:   br label %for.inc.1
; CHECK: for.inc.1:
; CHECK:   br i1 %exitcond.1, label %for.latch, label %for.inner
@a = hidden global [1 x i32] zeroinitializer, align 4
define i32 @test5() #0 {
entry:
  br label %for.outer

for.outer:
  %.sink16 = phi i32 [ 0, %entry ], [ %add, %for.latch ]
  br label %for.inner

for.inner:
  %inc8.sink15 = phi i32 [ 0, %for.outer ], [ %inc8, %for.inc ]
  br label %for.inner2

for.inner2:
  %l1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @a, i32 0, i32 0), align 4
  %tobool = icmp eq i32 %l1, 0
  br i1 %tobool, label %for.cond4, label %for.inc

for.cond4:
  %l0 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @a, i32 1, i32 0), align 4
  %tobool.1 = icmp eq i32 %l0, 0
  br i1 %tobool.1, label %for.cond4a, label %for.inc

for.cond4a:
  br label %for.inc

for.inc:
  %l2 = phi i32 [ 0, %for.inner2 ], [ 1, %for.cond4 ], [ 2, %for.cond4a ]
  %inc8 = add nuw nsw i32 %inc8.sink15, 1
  %exitcond = icmp eq i32 %inc8, 3
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %.lcssa = phi i32 [ %l2, %for.inc ]
  %conv11 = and i32 %.sink16, 255
  %add = add nuw nsw i32 %conv11, 4
  %cmp = icmp eq i32 %add, 8
  br i1 %cmp, label %for.end, label %for.outer

for.end:
  %.lcssa.lcssa = phi i32 [ %.lcssa, %for.latch ]
  ret i32 0
}




; CHECK-LABEL: test6
; Test odd uses of phi nodes
; CHECK: for.outer:
; CHECK:   br label %for.inner
; CHECK: for.inner:
; CHECK:   br i1 %exitcond.3, label %for.inner, label %for.latch
; CHECK: for.latch:
; CHECK:   br label %for.end
; CHECK: for.end:
; CHECK:   ret i32 0
@f = hidden global i32 0, align 4
define i32 @test6() #0 {
entry:
  %f.promoted10 = load i32, i32* @f, align 4, !tbaa !5
  br label %for.outer

for.outer:
  %p0 = phi i32 [ %f.promoted10, %entry ], [ 2, %for.latch ]
  %inc5.sink9 = phi i32 [ 2, %entry ], [ %inc5, %for.latch ]
  br label %for.inner

for.inner:
  %p1 = phi i32 [ %p0, %for.outer ], [ 2, %for.inner ]
  %inc.sink8 = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %inc = add nuw nsw i32 %inc.sink8, 1
  %exitcond = icmp ne i32 %inc, 7
  br i1 %exitcond, label %for.inner, label %for.latch

for.latch:
  %.lcssa = phi i32 [ %p1, %for.inner ]
  %inc5 = add nuw nsw i32 %inc5.sink9, 1
  %exitcond11 = icmp ne i32 %inc5, 7
  br i1 %exitcond11, label %for.outer, label %for.end

for.end:
  %.lcssa.lcssa = phi i32 [ %.lcssa, %for.latch ]
  %inc.lcssa.lcssa = phi i32 [ 7, %for.latch ]
  ret i32 0
}



; CHECK-LABEL: test7
; Has a positive dependency between two stores. Still valid.
; The negative dependecy is in unroll-and-jam-disabled.ll
; CHECK: for.outer:
; CHECK:   %i = phi i32 [ %add.3, %for.latch ], [ 0, %for.preheader.new ]
; CHECK:   %niter = phi i32 [ %unroll_iter, %for.preheader.new ], [ %niter.nsub.3, %for.latch ]
; CHECK:   br label %for.inner
; CHECK: for.latch:
; CHECK:   %add9.lcssa = phi i32 [ %add9, %for.inner ]
; CHECK:   %add9.lcssa.1 = phi i32 [ %add9.1, %for.inner ]
; CHECK:   %add9.lcssa.2 = phi i32 [ %add9.2, %for.inner ]
; CHECK:   %add9.lcssa.3 = phi i32 [ %add9.3, %for.inner ]
; CHECK:   br i1 %niter.ncmp.3, label %for.end.loopexit.unr-lcssa.loopexit, label %for.outer
; CHECK: for.inner:
; CHECK:   %sum = phi i32 [ 0, %for.outer ], [ %add9, %for.inner ]
; CHECK:   %j = phi i32 [ 0, %for.outer ], [ %add10, %for.inner ]
; CHECK:   %sum.1 = phi i32 [ 0, %for.outer ], [ %add9.1, %for.inner ]
; CHECK:   %j.1 = phi i32 [ 0, %for.outer ], [ %add10.1, %for.inner ]
; CHECK:   %sum.2 = phi i32 [ 0, %for.outer ], [ %add9.2, %for.inner ]
; CHECK:   %j.2 = phi i32 [ 0, %for.outer ], [ %add10.2, %for.inner ]
; CHECK:   %sum.3 = phi i32 [ 0, %for.outer ], [ %add9.3, %for.inner ]
; CHECK:   %j.3 = phi i32 [ 0, %for.outer ], [ %add10.3, %for.inner ]
; CHECK:   br i1 %exitcond.3, label %for.latch, label %for.inner
; CHECK: for.end.loopexit.unr-lcssa.loopexit:
define void @test7(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp128 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp128, %cmp
  br i1 %or.cond, label %for.preheader, label %for.end

for.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add, %for.latch ], [ 0, %for.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 0, i32* %arrayidx, align 4, !tbaa !5
  %add = add nuw i32 %i, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %add
  store i32 2, i32* %arrayidx2, align 4, !tbaa !5
  br label %for.inner

for.latch:
  store i32 %add9, i32* %arrayidx, align 4, !tbaa !5
  %exitcond30 = icmp eq i32 %add, %I
  br i1 %exitcond30, label %for.end, label %for.outer

for.inner:
  %sum = phi i32 [ 0, %for.outer ], [ %add9, %for.inner ]
  %j = phi i32 [ 0, %for.outer ], [ %add10, %for.inner ]
  %arrayidx7 = getelementptr inbounds i32, i32* %B, i32 %j
  %l1 = load i32, i32* %arrayidx7, align 4, !tbaa !5
  %add9 = add i32 %l1, %sum
  %add10 = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %add10, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.end:
  ret void
}



; CHECK-LABEL: test8
; Same as test7 with an extra outer loop nest
; CHECK: for.outest:
; CHECK:   br label %for.outer
; CHECK: for.outer:
; CHECK:   %i = phi i32 [ %add.3, %for.latch ], [ 0, %for.outest.new ]
; CHECK:   %niter = phi i32 [ %unroll_iter, %for.outest.new ], [ %niter.nsub.3, %for.latch ]
; CHECK:   br label %for.inner
; CHECK: for.inner:
; CHECK:   %sum = phi i32 [ 0, %for.outer ], [ %add9, %for.inner ]
; CHECK:   %j = phi i32 [ 0, %for.outer ], [ %add10, %for.inner ]
; CHECK:   %sum.1 = phi i32 [ 0, %for.outer ], [ %add9.1, %for.inner ]
; CHECK:   %j.1 = phi i32 [ 0, %for.outer ], [ %add10.1, %for.inner ]
; CHECK:   %sum.2 = phi i32 [ 0, %for.outer ], [ %add9.2, %for.inner ]
; CHECK:   %j.2 = phi i32 [ 0, %for.outer ], [ %add10.2, %for.inner ]
; CHECK:   %sum.3 = phi i32 [ 0, %for.outer ], [ %add9.3, %for.inner ]
; CHECK:   %j.3 = phi i32 [ 0, %for.outer ], [ %add10.3, %for.inner ]
; CHECK:   br i1 %exitcond.3, label %for.latch, label %for.inner
; CHECK: for.latch:
; CHECK:   %add9.lcssa = phi i32 [ %add9, %for.inner ]
; CHECK:   %add9.lcssa.1 = phi i32 [ %add9.1, %for.inner ]
; CHECK:   %add9.lcssa.2 = phi i32 [ %add9.2, %for.inner ]
; CHECK:   %add9.lcssa.3 = phi i32 [ %add9.3, %for.inner ]
; CHECK:   br i1 %niter.ncmp.3, label %for.cleanup.unr-lcssa.loopexit, label %for.outer
; CHECK: for.cleanup.epilog-lcssa:
; CHECK:   br label %for.cleanup
; CHECK: for.cleanup:
; CHECK:   br i1 %exitcond41, label %for.end.loopexit, label %for.outest
; CHECK: for.end.loopexit:
; CHECK:   br label %for.end
define void @test8(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) #0 {
entry:
  %cmp = icmp eq i32 %J, 0
  %cmp336 = icmp eq i32 %I, 0
  %or.cond = or i1 %cmp, %cmp336
  br i1 %or.cond, label %for.end, label %for.preheader

for.preheader:
  br label %for.outest

for.outest:
  %x.038 = phi i32 [ %inc, %for.cleanup ], [ 0, %for.preheader ]
  br label %for.outer

for.outer:
  %i = phi i32 [ %add, %for.latch ], [ 0, %for.outest ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 0, i32* %arrayidx, align 4, !tbaa !5
  %add = add nuw i32 %i, 1
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %add
  store i32 2, i32* %arrayidx6, align 4, !tbaa !5
  br label %for.inner

for.inner:
  %sum = phi i32 [ 0, %for.outer ], [ %add9, %for.inner ]
  %j = phi i32 [ 0, %for.outer ], [ %add10, %for.inner ]
  %arrayidx11 = getelementptr inbounds i32, i32* %B, i32 %j
  %l1 = load i32, i32* %arrayidx11, align 4, !tbaa !5
  %add9 = add i32 %l1, %sum
  %add10 = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %add10, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  store i32 %add9, i32* %arrayidx, align 4, !tbaa !5
  %exitcond39 = icmp eq i32 %add, %I
  br i1 %exitcond39, label %for.cleanup, label %for.outer

for.cleanup:
  %inc = add nuw nsw i32 %x.038, 1
  %exitcond41 = icmp eq i32 %inc, 5
  br i1 %exitcond41, label %for.end, label %for.outest


for.end:
  ret void
}

; CHECK-LABEL: test9
; Same as test1 with tbaa, not noalias
; CHECK: for.outer:
; CHECK:   %i.us = phi i32 [ %add8.us.3, %for.latch ], [ 0, %for.outer.preheader.new ]
; CHECK:   %niter = phi i32 [ %unroll_iter, %for.outer.preheader.new ], [ %niter.nsub.3, %for.latch ]
; CHECK:   br label %for.inner
; CHECK: for.inner:
; CHECK:   %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
; CHECK:   %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
; CHECK:   %j.us.1 = phi i32 [ 0, %for.outer ], [ %inc.us.1, %for.inner ]
; CHECK:   %sum1.us.1 = phi i32 [ 0, %for.outer ], [ %add.us.1, %for.inner ]
; CHECK:   %j.us.2 = phi i32 [ 0, %for.outer ], [ %inc.us.2, %for.inner ]
; CHECK:   %sum1.us.2 = phi i32 [ 0, %for.outer ], [ %add.us.2, %for.inner ]
; CHECK:   %j.us.3 = phi i32 [ 0, %for.outer ], [ %inc.us.3, %for.inner ]
; CHECK:   %sum1.us.3 = phi i32 [ 0, %for.outer ], [ %add.us.3, %for.inner ]
; CHECK:   br i1 %exitcond.3, label %for.latch, label %for.inner
; CHECK: for.latch:
; CHECK:   %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
; CHECK:   %add.us.lcssa.1 = phi i32 [ %add.us.1, %for.inner ]
; CHECK:   %add.us.lcssa.2 = phi i32 [ %add.us.2, %for.inner ]
; CHECK:   %add.us.lcssa.3 = phi i32 [ %add.us.3, %for.inner ]
; CHECK:   br i1 %niter.ncmp.3, label %for.end.loopexit.unr-lcssa.loopexit, label %for.outer
; CHECK: for.end.loopexit.unr-lcssa.loopexit:
define void @test9(i32 %I, i32 %J, i32* nocapture %A, i16* nocapture readonly %B) #0 {
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
  %arrayidx.us = getelementptr inbounds i16, i16* %B, i32 %j.us
  %0 = load i16, i16* %arrayidx.us, align 4, !tbaa !9
  %sext = sext i16 %0 to i32
  %add.us = add i32 %sext, %sum1.us
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



; CHECK-LABEL: test10
; Be careful not to incorrectly update the exit phi nodes
; CHECK: %dec19.lcssa.lcssa.lcssa.ph.ph = phi i64 [ 0, %for.inc24 ]
%struct.a = type { i64 }
@g = common global %struct.a zeroinitializer, align 8
@c = common global [1 x i8] zeroinitializer, align 1
; Function Attrs: noinline norecurse nounwind uwtable
define signext i16 @test10(i32 %k) #0 {
entry:
  %0 = load i8, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @c, i64 0, i64 0), align 1
  %tobool9 = icmp eq i8 %0, 0
  %tobool13 = icmp ne i32 %k, 0
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc24
  %storemerge82 = phi i64 [ 0, %entry ], [ %inc25, %for.inc24 ]
  br label %for.body2

for.body2:                                        ; preds = %for.body, %for.inc21
  %storemerge2881 = phi i64 [ 4, %for.body ], [ %dec22, %for.inc21 ]
  br i1 %tobool9, label %for.body2.split, label %for.body2.split.us

for.body2.split.us:                               ; preds = %for.body2
  br i1 %tobool13, label %for.inc21, label %for.inc21.loopexit83

for.body2.split:                                  ; preds = %for.body2
  br i1 %tobool13, label %for.inc21, label %for.inc21.loopexit85

for.inc21.loopexit83:                             ; preds = %for.body2.split.us
  %storemerge31.us37.lcssa.lcssa = phi i64 [ 0, %for.body2.split.us ]
  br label %for.inc21

for.inc21.loopexit85:                             ; preds = %for.body2.split
  %storemerge31.lcssa.lcssa87 = phi i64 [ 0, %for.body2.split ]
  %storemerge30.lcssa.lcssa86 = phi i32 [ 0, %for.body2.split ]
  br label %for.inc21

for.inc21:                                        ; preds = %for.body2.split, %for.body2.split.us, %for.inc21.loopexit85, %for.inc21.loopexit83
  %storemerge31.lcssa.lcssa = phi i64 [ %storemerge31.us37.lcssa.lcssa, %for.inc21.loopexit83 ], [ %storemerge31.lcssa.lcssa87, %for.inc21.loopexit85 ], [ 4, %for.body2.split.us ], [ 4, %for.body2.split ]
  %storemerge30.lcssa.lcssa = phi i32 [ 0, %for.inc21.loopexit83 ], [ %storemerge30.lcssa.lcssa86, %for.inc21.loopexit85 ], [ 0, %for.body2.split.us ], [ 0, %for.body2.split ]
  %dec22 = add nsw i64 %storemerge2881, -1
  %tobool = icmp eq i64 %dec22, 0
  br i1 %tobool, label %for.inc24, label %for.body2

for.inc24:                                        ; preds = %for.inc21
  %storemerge31.lcssa.lcssa.lcssa = phi i64 [ %storemerge31.lcssa.lcssa, %for.inc21 ]
  %storemerge30.lcssa.lcssa.lcssa = phi i32 [ %storemerge30.lcssa.lcssa, %for.inc21 ]
  %inc25 = add nuw nsw i64 %storemerge82, 1
  %exitcond = icmp ne i64 %inc25, 5
  br i1 %exitcond, label %for.body, label %for.end26

for.end26:                                        ; preds = %for.inc24
  %dec19.lcssa.lcssa.lcssa = phi i64 [ 0, %for.inc24 ]
  %storemerge31.lcssa.lcssa.lcssa.lcssa = phi i64 [ %storemerge31.lcssa.lcssa.lcssa, %for.inc24 ]
  %storemerge30.lcssa.lcssa.lcssa.lcssa = phi i32 [ %storemerge30.lcssa.lcssa.lcssa, %for.inc24 ]
  store i64 %dec19.lcssa.lcssa.lcssa, i64* getelementptr inbounds (%struct.a, %struct.a* @g, i64 0, i32 0), align 8
  ret i16 0
}


attributes #0 = { "target-cpu"="cortex-m33" }

!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"short", !7, i64 0}
