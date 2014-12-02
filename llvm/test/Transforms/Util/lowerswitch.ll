; RUN: opt -lowerswitch -S < %s | FileCheck %s

; Test that we don't crash and have a different basic block for each incoming edge.
define void @test0() {
; CHECK-LABEL: @test0
; CHECK: %merge = phi i64 [ 1, %BB3 ], [ 0, %NewDefault ], [ 0, %NodeBlock5 ], [ 0, %LeafBlock1 ]
BB1:
  switch i32 undef, label %BB2 [
    i32 3, label %BB2
    i32 5, label %BB2
    i32 0, label %BB3
    i32 2, label %BB3
    i32 4, label %BB3
  ]

BB2:
  %merge = phi i64 [ 1, %BB3 ], [ 0, %BB1 ], [ 0, %BB1 ], [ 0, %BB1 ]
  ret void

BB3:
  br label %BB2
}

; Test switch cases that are merged into a single case during lowerswitch
; (take 84 and 85 below) - check that the number of incoming phi values match
; the number of branches.
define void @test1() {
; CHECK-LABEL: @test1
entry:
  br label %bb1

bb1:
  switch i32 undef, label %bb1 [
    i32 84, label %bb3
    i32 85, label %bb3
    i32 86, label %bb2
    i32 78, label %exit
    i32 99, label %bb3
  ]

bb2:
  br label %bb3

bb3:
; CHECK-LABEL: bb3
; CHECK: %tmp = phi i32 [ 1, %NodeBlock ], [ 0, %bb2 ], [ 1, %LeafBlock3 ]
  %tmp = phi i32 [ 1, %bb1 ], [ 0, %bb2 ], [ 1, %bb1 ], [ 1, %bb1 ]
; CHECK-NEXT: %tmp2 = phi i32 [ 2, %NodeBlock ], [ 5, %bb2 ], [ 2, %LeafBlock3 ]
  %tmp2 = phi i32 [ 2, %bb1 ], [ 2, %bb1 ], [ 5, %bb2 ], [ 2, %bb1 ]
  br label %exit

exit:
  ret void
}
