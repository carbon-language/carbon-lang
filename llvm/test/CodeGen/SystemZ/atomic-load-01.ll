; Test 8-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This is just a placeholder to make sure that loads are handled.
; The CS-based sequence is probably far too conservative.
define i8 @f1(i8 *%src) {
; CHECK-LABEL: f1:
; CHECK: cs
; CHECK: br %r14
  %val = load atomic i8 *%src seq_cst, align 1
  ret i8 %val
}
