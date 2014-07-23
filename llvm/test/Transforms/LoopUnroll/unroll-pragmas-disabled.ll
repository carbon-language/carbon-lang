; RUN: opt < %s -loop-unroll -S | FileCheck %s
;
; Verify that the unrolling pass removes existing unroll count metadata
; and adds a disable unrolling node after unrolling is complete.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; #pragma clang loop  vectorize(enable) unroll_count(4) vectorize_width(8)
;
; Unroll count metadata should be replaced with unroll(disable).  Vectorize
; metadata should be untouched.
;
; CHECK-LABEL: @loop1(
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
define void @loop1(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %a, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:                                          ; preds = %for.body
  ret void
}
!1 = metadata !{metadata !1, metadata !2, metadata !3, metadata !4}
!2 = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
!3 = metadata !{metadata !"llvm.loop.unroll.count", i32 4}
!4 = metadata !{metadata !"llvm.loop.vectorize.width", i32 8}

; #pragma clang loop unroll(full)
;
; An unroll disable metadata node is only added for the unroll count case.
; In this case, the loop has a full unroll metadata but can't be fully unrolled
; because the trip count is dynamic.  The full unroll metadata should remain
; after unrolling.
;
; CHECK-LABEL: @loop2(
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
define void @loop2(i32* nocapture %a, i32 %b) {
entry:
  %cmp3 = icmp sgt i32 %b, 0
  br i1 %cmp3, label %for.body, label %for.end, !llvm.loop !5

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %a, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %b
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !5

for.end:                                          ; preds = %for.body, %entry
  ret void
}
!5 = metadata !{metadata !5, metadata !6}
!6 = metadata !{metadata !"llvm.loop.unroll.full"}

; #pragma clang loop unroll(disable)
;
; Unroll metadata should not change.
;
; CHECK-LABEL: @loop3(
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
define void @loop3(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %a, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !7

for.end:                                          ; preds = %for.body
  ret void
}
!7 = metadata !{metadata !7, metadata !8}
!8 = metadata !{metadata !"llvm.loop.unroll.disable"}

; CHECK: ![[LOOP_1]] = metadata !{metadata ![[LOOP_1]], metadata ![[VEC_ENABLE:.*]], metadata ![[WIDTH_8:.*]], metadata ![[UNROLL_DISABLE:.*]]}
; CHECK: ![[VEC_ENABLE]] = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
; CHECK: ![[WIDTH_8]] = metadata !{metadata !"llvm.loop.vectorize.width", i32 8}
; CHECK: ![[UNROLL_DISABLE]] = metadata !{metadata !"llvm.loop.unroll.disable"}
; CHECK: ![[LOOP_2]] = metadata !{metadata ![[LOOP_2]], metadata ![[UNROLL_FULL:.*]]}
; CHECK: ![[UNROLL_FULL]] = metadata !{metadata !"llvm.loop.unroll.full"}
; CHECK: ![[LOOP_3]] = metadata !{metadata ![[LOOP_3]], metadata ![[UNROLL_DISABLE:.*]]}
