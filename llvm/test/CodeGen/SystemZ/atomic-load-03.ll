; Test 32-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that loads are handled.
; Using CS is probably too conservative.
define i32 @f1(i32 %dummy, i32 *%src) {
; CHECK: f1:
; CHECK: lhi %r2, 0
; CHECK: cs %r2, %r2, 0(%r3)
; CHECK: br %r14
  %val = load atomic i32 *%src seq_cst, align 4
  ret i32 %val
}
