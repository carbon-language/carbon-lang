; Test 32-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that stores are handled.
; Using CS is probably too conservative.
define void @f1(i32 %val, i32 *%src) {
; CHECK-LABEL: f1:
; CHECK: l %r0, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: cs %r0, %r2, 0(%r3)
; CHECK: jlh [[LABEL]]
; CHECK: br %r14
  store atomic i32 %val, i32 *%src seq_cst, align 4
  ret void
}
