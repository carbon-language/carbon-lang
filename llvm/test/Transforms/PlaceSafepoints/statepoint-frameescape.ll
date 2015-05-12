; RUN: opt %s -S -place-safepoints | FileCheck %s

declare void @llvm.frameescape(...)

; Do we insert the entry safepoint after the frameescape intrinsic?
define void @parent() gc "statepoint-example" {
; CHECK-LABEL: @parent
entry:
; CHECK-LABEL: entry
; CHECK-NEXT: alloca
; CHECK-NEXT: frameescape
; CHECK-NEXT: statepoint
  %ptr = alloca i32
  call void (...) @llvm.frameescape(i32* %ptr)
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