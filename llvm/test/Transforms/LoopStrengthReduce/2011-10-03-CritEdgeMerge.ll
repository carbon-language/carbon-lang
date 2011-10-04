; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; Test LSR's use of SplitCriticalEdge during phi rewriting.
; Verify that identical edges are merged. rdar://problem/6453893

target triple = "x86-apple-darwin"

; CHECK: @test
; CHECK: bb89:
; CHECK: phi i8* [ %lsr.iv.next1, %bbA.bb89_crit_edge ], [ %lsr.iv.next1, %bbB.bb89_crit_edge ]{{$}}

define i8* @test() {
entry:
  br label %loop

loop:
  %rec = phi i32 [ %next, %loop ], [ 0, %entry ]
  %next = add i32 %rec, 1
  %tmp75 = getelementptr i8* null, i32 %next
  br i1 false, label %loop, label %loopexit

loopexit:
  br i1 false, label %bbA, label %bbB

bbA:
  switch i32 0, label %bb89 [
    i32 47, label %bb89
    i32 58, label %bb89
  ]

bbB:
  switch i8 0, label %bb89 [
    i8 47, label %bb89
    i8 58, label %bb89
  ]

bb89:
  %tmp75phi = phi i8* [ %tmp75, %bbA ], [ %tmp75, %bbA ], [ %tmp75, %bbA ], [ %tmp75, %bbB ], [ %tmp75, %bbB ], [ %tmp75, %bbB ]
  br label %exit

exit:
  ret i8* %tmp75phi
}
