; Test 64-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that loads are handled.
; Using CSG is probably too conservative.
define i64 @f1(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f1:
; CHECK: lghi %r2, 0
; CHECK: csg %r2, %r2, 0(%r3)
; CHECK: br %r14
  %val = load atomic i64 *%src seq_cst, align 8
  ret i64 %val
}
