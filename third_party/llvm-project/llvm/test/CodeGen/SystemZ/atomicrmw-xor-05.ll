; Test 32-bit atomic ORs, z196 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check OR of a variable.
define i32 @f1(i32 %dummy, i32 *%src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: lax %r2, %r4, 0(%r3)
; CHECK: br %r14
  %res = atomicrmw xor i32 *%src, i32 %b seq_cst
  ret i32 %res
}

; Check OR of 1, which needs a temporary.
define i32 @f2(i32 %dummy, i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: lhi [[TMP:%r[0-5]]], 1
; CHECK: lax %r2, [[TMP]], 0(%r3)
; CHECK: br %r14
  %res = atomicrmw xor i32 *%src, i32 1 seq_cst
  ret i32 %res
}

; Check the high end of the LAX range.
define i32 @f3(i32 %dummy, i32 *%src, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: lax %r2, %r4, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i32 131071
  %res = atomicrmw xor i32 *%ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word up, which needs separate address logic.
define i32 @f4(i32 %dummy, i32 *%src, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: lax %r2, %r4, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i32 131072
  %res = atomicrmw xor i32 *%ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the low end of the LAX range.
define i32 @f5(i32 %dummy, i32 *%src, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: lax %r2, %r4, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i32 -131072
  %res = atomicrmw xor i32 *%ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word down, which needs separate address logic.
define i32 @f6(i32 %dummy, i32 *%src, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, -524292
; CHECK: lax %r2, %r4, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i32 -131073
  %res = atomicrmw xor i32 *%ptr, i32 %b seq_cst
  ret i32 %res
}
