; RUN: opt < %s -loop-unroll -S | FileCheck %s
;
; Verify that the unrolling pass removes existing loop unrolling metadata
; and adds a disable unrolling node after unrolling is complete.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; #pragma clang loop  vectorize(enable) unroll(enable) unroll_count(4) vectorize_width(8)
;
; Unroll metadata should be replaces with unroll(disable).  Vectorize
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
!1 = metadata !{metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!2 = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
!3 = metadata !{metadata !"llvm.loop.unroll.enable", i1 true}
!4 = metadata !{metadata !"llvm.loop.unroll.count", i32 4}
!5 = metadata !{metadata !"llvm.loop.vectorize.width", i32 8}

; #pragma clang loop unroll(disable)
;
; Unroll metadata should not change.
;
; CHECK-LABEL: @loop2(
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
define void @loop2(i32* nocapture %a) {
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !6

for.end:                                          ; preds = %for.body
  ret void
}
!6 = metadata !{metadata !6, metadata !7}
!7 = metadata !{metadata !"llvm.loop.unroll.enable", i1 false}

; CHECK: ![[LOOP_1]] = metadata !{metadata ![[LOOP_1]], metadata ![[VEC_ENABLE:.*]], metadata ![[WIDTH_8:.*]], metadata ![[UNROLL_DISABLE:.*]]}
; CHECK: ![[VEC_ENABLE]] = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
; CHECK: ![[WIDTH_8]] = metadata !{metadata !"llvm.loop.vectorize.width", i32 8}
; CHECK: ![[UNROLL_DISABLE]] = metadata !{metadata !"llvm.loop.unroll.enable", i1 false}
; CHECK: ![[LOOP_2]] = metadata !{metadata ![[LOOP_2]], metadata ![[UNROLL_DISABLE:.*]]}
