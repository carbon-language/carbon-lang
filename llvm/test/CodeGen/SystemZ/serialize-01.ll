; Test serialization instructions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | \
; RUN:   FileCheck %s -check-prefix=CHECK-FULL
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | \
; RUN:   FileCheck %s -check-prefix=CHECK-FAST

; Check that volatile loads produce a serialisation.
define i32 @f1(i32 *%src) {
; CHECK-FULL-LABEL: f1:
; CHECK-FULL: bcr 15, %r0
; CHECK-FULL: l %r2, 0(%r2)
; CHECK-FULL: br %r14
;
; CHECK-FAST-LABEL: f1:
; CHECK-FAST: bcr 14, %r0
; CHECK-FAST: l %r2, 0(%r2)
; CHECK-FAST: br %r14
  %val = load volatile i32 *%src
  ret i32 %val
}
