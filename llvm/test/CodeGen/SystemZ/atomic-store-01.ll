; Test 8-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that stores are handled.
; The CS-based sequence is probably far too conservative.
define void @f1(i8 %val, i8 *%src) {
; CHECK: f1:
; CHECK: cs
; CHECK: br %r14
  store atomic i8 %val, i8 *%src seq_cst, align 1
  ret void
}
