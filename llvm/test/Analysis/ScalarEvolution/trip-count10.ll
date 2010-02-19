; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; Trip counts with trivial exit conditions.

; CHECK: Determining loop execution counts for: @a
; CHECK: Loop %loop: Unpredictable backedge-taken count. 
; CHECK: Loop %loop: Unpredictable max backedge-taken count.

; CHECK: Determining loop execution counts for: @b
; CHECK: Loop %loop: backedge-taken count is false
; CHECK: Loop %loop: max backedge-taken count is false

; CHECK: Determining loop execution counts for: @c
; CHECK: Loop %loop: backedge-taken count is false
; CHECK: Loop %loop: max backedge-taken count is false

; CHECK: Determining loop execution counts for: @d
; CHECK: Loop %loop: Unpredictable backedge-taken count. 
; CHECK: Loop %loop: Unpredictable max backedge-taken count. 

define void @a(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 false, label %return, label %loop

return:
  ret void
}
define void @b(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 true, label %return, label %loop

return:
  ret void
}
define void @c(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 false, label %loop, label %return

return:
  ret void
}
define void @d(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 true, label %loop, label %return

return:
  ret void
}
