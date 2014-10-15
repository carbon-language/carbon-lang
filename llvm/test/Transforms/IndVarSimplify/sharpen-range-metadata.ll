;; RUN: opt -S < %s -indvars | FileCheck %s

;; Check if llvm can narrow !range metadata based on loop entry
;; predicates.

declare void @abort()

define i1 @bounded_below(i32* nocapture readonly %buffer) {
entry:
  %length = load i32* %buffer, !range !0
  %entry.pred = icmp eq i32 %length, 0
  br i1 %entry.pred, label %abort, label %loop.preheader

loop.preheader:
  br label %loop

loop:
  %idx = phi i32 [ %idx.inc, %loop.next ], [ 0, %loop.preheader ]
  %oob.pred = icmp slt i32 %idx, %length
  br i1 %oob.pred, label %loop.next, label %oob
; CHECK: br i1 true, label %loop.next, label %oob

loop.next:
  %idx.inc = add i32 %idx, 1
  %exit.pred = icmp slt i32 %idx.inc, %length
  br i1 %exit.pred, label %loop, label %abort.loopexit

abort.loopexit:
  br label %abort

abort:
  ret i1 false

oob:
  tail call void @abort()
  ret i1 false
}

!0 = metadata !{i32 0, i32 100}
