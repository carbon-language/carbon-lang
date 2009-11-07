; RUN: opt %s -tailcallelim -S | FileCheck %s

define i64 @fib(i64 %n) nounwind readnone {
; CHECK: @fib
entry:
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i64 [ %n, %entry ], [ %3, %bb1 ]
; CHECK: %n.tr = phi i64 [ %n, %entry ], [ %2, %bb1 ]
  switch i64 %n, label %bb1 [
; CHECK: switch i64 %n.tr, label %bb1 [
    i64 0, label %bb2
    i64 1, label %bb2
  ]

bb1:
; CHECK: bb1:
  %0 = add i64 %n, -1
; CHECK: %0 = add i64 %n.tr, -1
  %1 = tail call i64 @fib(i64 %0) nounwind
; CHECK: %1 = tail call i64 @fib(i64 %0)
  %2 = add i64 %n, -2
; CHECK: %2 = add i64 %n.tr, -2
  %3 = tail call i64 @fib(i64 %2) nounwind
; CHECK-NOT: tail call i64 @fib
  %4 = add nsw i64 %3, %1
; CHECK: add nsw i64 %accumulator.tr, %1
  ret i64 %4
; CHECK: br label %tailrecurse

bb2:
; CHECK: bb2:
  ret i64 %n
; CHECK: ret i64 %accumulator.tr
}
