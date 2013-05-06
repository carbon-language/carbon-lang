; Test the stackrestore builtin.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.stackrestore(i8 *)

; we should use a frame pointer and tear down the frame based on %r11
; rather than %r15.
define void @f1(i8 *%src) {
; CHECK: f1:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: lgr %r11, %r15
; CHECK: lgr %r15, %r2
; CHECK: lmg %r11, %r15, 88(%r11)
; CHECK: br %r14
  call void @llvm.stackrestore(i8 *%src)
  ret void
}
