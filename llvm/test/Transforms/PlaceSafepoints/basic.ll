; RUN: opt %s -S -place-safepoints | FileCheck %s


; Do we insert a simple entry safepoint?
define void @test_entry() gc "statepoint-example" {
; CHECK-LABEL: @test_entry
entry:
; CHECK-LABEL: entry
; CHECK: statepoint
  ret void
}

; Do we insert a backedge safepoint in a statically
; infinite loop?
define void @test_backedge() gc "statepoint-example" {
; CHECK-LABEL: test_backedge
entry:
; CHECK-LABEL: entry
; This statepoint is technically not required, but we don't exploit that yet.
; CHECK: statepoint
  br label %other

; CHECK-LABEL: other
; CHECK: statepoint
other:
  call void undef()
  br label %other
}

; Check that we remove an unreachable block rather than trying
; to insert a backedge safepoint
define void @test_unreachable() gc "statepoint-example" {
; CHECK-LABEL: test_unreachable
entry:
; CHECK-LABEL: entry
; CHECK: statepoint
  ret void

; CHECK-NOT: other
; CHECK-NOT: statepoint
other:
  br label %other
}

declare void @foo()

; Do we turn a call into it's own statepoint
define void @test_simple_call() gc "statepoint-example" {
; CHECK-LABEL: test_simple_call
entry:
  br label %other
other:
; CHECK-LABEL: other
; CHECK: statepoint 
  call void @foo()
  ret void
}


; This function is inlined when inserting a poll.  To avoid recursive 
; issues, make sure we don't place safepoints in it.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
; CHECK-LABEL: entry
; CHECK-NEXT: do_safepoint
; CHECK-NEXT: ret void 
entry:
  call void @do_safepoint()
  ret void
}
