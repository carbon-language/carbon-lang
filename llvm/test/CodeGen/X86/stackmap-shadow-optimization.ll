; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -disable-fp-elim | FileCheck %s

; Check that the X86 stackmap shadow optimization is only outputting a 1-byte
; nop here. 8-bytes are requested, but 7 are covered by the code for the call to
; bar, the frame teardown and the return.
define void @shadow_optimization_test() {
entry:
; CHECK-LABEL:  shadow_optimization_test:
; CHECK:        callq   _bar
; CHECK-NOT:    nop
; CHECK:        callq   _bar
; CHECK:        retq
; CHECK:        nop
  call void @bar()
  tail call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 0, i32 8)
  call void @bar()
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @bar()