; If there's a call in the loop which dominates the backedge, we 
; don't need a safepoint poll (since the callee must contain a 
; poll test).
;; RUN: opt < %s -place-safepoints -S | FileCheck %s

declare void @foo()

define void @test1() gc "statepoint-example" {
; CHECK-LABEL: test1

entry:
; CHECK-LABEL: entry
; CHECK: statepoint
  br label %loop

loop:
; CHECK-LABEL: loop
; CHECK: @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @foo
; CHECK-NOT: statepoint
  call void @foo()
  br label %loop
}

; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}
