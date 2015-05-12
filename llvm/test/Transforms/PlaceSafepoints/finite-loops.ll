; Tests to ensure that we are not placing backedge safepoints in
; loops which are clearly finite.
;; RUN: opt %s -place-safepoints -S | FileCheck %s


; A simple counted loop with trivially known range
define void @test1(i32) gc "statepoint-example" {
; CHECK-LABEL: test1
; CHECK-LABEL: entry
; CHECK: statepoint
; CHECK-LABEL: loop
; CHECK-NOT: statepoint

entry:
  br label %loop

loop:
  %counter = phi i32 [ 0 , %entry ], [ %counter.inc , %loop ]
  %counter.inc = add i32 %counter, 1
  %counter.cmp = icmp slt i32 %counter.inc, 16
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}

; The same counted loop, but with an unknown early exit
define void @test2(i32) gc "statepoint-example" {
; CHECK-LABEL: test2
; CHECK-LABEL: entry
; CHECK: statepoint
; CHECK-LABEL: loop
; CHECK-NOT: statepoint

entry:
  br label %loop

loop:
  %counter = phi i32 [ 0 , %entry ], [ %counter.inc , %continue ]
  %counter.inc = add i32 %counter, 1
  %counter.cmp = icmp slt i32 %counter.inc, 16
  br i1 undef, label %continue, label %exit

continue:
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}

; The range is a 8 bit value and we can't overflow
define void @test3(i8 %upper) gc "statepoint-example" {
; CHECK-LABEL: test3
; CHECK-LABEL: entry
; CHECK: statepoint
; CHECK-LABEL: loop
; CHECK-NOT: statepoint

entry:
  br label %loop

loop:
  %counter = phi i8 [ 0 , %entry ], [ %counter.inc , %loop ]
  %counter.inc = add nsw i8 %counter, 1
  %counter.cmp = icmp slt i8 %counter.inc, %upper
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}


; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}