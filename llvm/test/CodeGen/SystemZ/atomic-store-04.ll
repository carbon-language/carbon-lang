; Test 64-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that stores are handled.
; Using CS is probably too conservative.
define void @f1(i64 %val, i64 *%src) {
; CHECK-LABEL: f1:
; CHECK: lg %r0, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: csg %r0, %r2, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  store atomic i64 %val, i64 *%src seq_cst, align 8
  ret void
}
