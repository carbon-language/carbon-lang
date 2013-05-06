; Test 16-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that stores are handled.
; The CS-based sequence is probably far too conservative.
define void @f1(i16 %val, i16 *%src) {
; CHECK: f1:
; CHECK: cs
; CHECK: br %r14
  store atomic i16 %val, i16 *%src seq_cst, align 2
  ret void
}
