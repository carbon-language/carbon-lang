; RUN: opt -lowerswitch -S < %s | FileCheck %s

; Test that we don't crash and have a different basic block for each incoming edge.
define void @test_lower_switch() {
; CHECK-LABEL: @test_lower_switch
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
