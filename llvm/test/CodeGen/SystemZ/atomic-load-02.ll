; Test 16-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that loads are handled.
; The CS-based sequence is probably far too conservative.
define i16 @f1(i16 *%src) {
; CHECK: f1:
; CHECK: cs
; CHECK: br %r14
  %val = load atomic i16 *%src seq_cst, align 2
  ret i16 %val
}
