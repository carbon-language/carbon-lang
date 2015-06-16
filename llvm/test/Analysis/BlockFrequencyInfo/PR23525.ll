; RUN: opt < %s -analyze -block-freq | FileCheck %s

@g = global i32 0, align 4

; Function Attrs: inlinehint noinline nounwind uwtable
define i32 @_Z8hot_loopi(i32 %n) !prof !1 {
entry:
  %div = sdiv i32 %n, 2
  %rem12 = and i32 %n, 1
  %cmp = icmp eq i32 %rem12, 0
  br i1 %cmp, label %Next, label %for.cond, !prof !2

; CHECK: - for.cond: float = 25.85{{[0-9]*}}, int = 206
for.cond:                                         ; preds = %entry, %for.inc
  %i.0 = phi i32 [ %inc, %for.inc ], [ %div, %entry ]
  %cmp1 = icmp slt i32 %i.0, %n
  br i1 %cmp1, label %for.body, label %for.end, !prof !3, !llvm.loop !4

; CHECK: - for.body: float = 24.52, int = 196
for.body:                                         ; preds = %for.cond
  %rem213 = and i32 %i.0, 1
  %cmp3 = icmp eq i32 %rem213, 0
  br i1 %cmp3, label %if.then.4, label %Next, !prof !6

; CHECK: - if.then.4: float = 12.26{{[0-9]*}}, int = 98
if.then.4:                                        ; preds = %for.body
  %0 = load i32, i32* @g, align 4, !tbaa !7
  %mul = shl nsw i32 %0, 1
  br label %for.inc

; CHECK: - Next: float = 12.41{{[0-9]*}}, int = 99
Next:                                             ; preds = %for.body, %entry
  %i.1 = phi i32 [ %div, %entry ], [ %i.0, %for.body ]
  %1 = load i32, i32* @g, align 4, !tbaa !7
  %add = add nsw i32 %1, %n
  br label %for.inc

; CHECK: - for.inc: float = 38.28{{[0-9]*}}, int = 306
for.inc:                                          ; preds = %if.then.4, %Next
  %storemerge = phi i32 [ %add, %Next ], [ %mul, %if.then.4 ]
  %i.2 = phi i32 [ %i.1, %Next ], [ %i.0, %if.then.4 ]
  store i32 %storemerge, i32* @g, align 4, !tbaa !7
  %inc = add nsw i32 %i.2, 1
  br label %for.cond

; CHECK: - for.end: float = 1.0, int = 8
for.end:                                          ; preds = %for.cond
  %2 = load i32, i32* @g, align 4, !tbaa !7
  ret i32 %2
}

; Function Attrs: nounwind uwtable
define i32 @main() !prof !11 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 0

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  %call = tail call i32 @_Z8hot_loopi(i32 %i.04)
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !prof !12
}


!1 = !{!"function_entry_count", i64 99}
!2 = !{!"branch_weights", i32 50, i32 51}
!3 = !{!"branch_weights", i32 2452, i32 100}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.unroll.disable"}
!6 = !{!"branch_weights", i32 1227, i32 1226}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"function_entry_count", i64 1}
!12 = !{!"branch_weights", i32 2, i32 100}
