; Test 64-bit compare and swap.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check CSG without a displacement.
define i64 @f1(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK: f1:
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %val = cmpxchg i64 *%src, i64 %cmp, i64 %swap seq_cst
  ret i64 %val
}

; Check the high end of the aligned CSG range.
define i64 @f2(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK: f2:
; CHECK: csg %r2, %r3, 524280(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65535
  %val = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst
  ret i64 %val
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f3(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK: f3:
; CHECK: agfi %r4, 524288
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65536
  %val = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst
  ret i64 %val
}

; Check the high end of the negative aligned CSG range.
define i64 @f4(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK: f4:
; CHECK: csg %r2, %r3, -8(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -1
  %val = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst
  ret i64 %val
}

; Check the low end of the CSG range.
define i64 @f5(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK: f5:
; CHECK: csg %r2, %r3, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65536
  %val = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst
  ret i64 %val
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK: f6:
; CHECK: agfi %r4, -524296
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65537
  %val = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst
  ret i64 %val
}

; Check that CSG does not allow an index.
define i64 @f7(i64 %cmp, i64 %swap, i64 %src, i64 %index) {
; CHECK: f7:
; CHECK: agr %r4, %r5
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %ptr = inttoptr i64 %add1 to i64 *
  %val = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst
  ret i64 %val
}

; Check that a constant %cmp value is loaded into a register first.
define i64 @f8(i64 %dummy, i64 %swap, i64 *%ptr) {
; CHECK: f8:
; CHECK: lghi %r2, 1001
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %val = cmpxchg i64 *%ptr, i64 1001, i64 %swap seq_cst
  ret i64 %val
}

; Check that a constant %swap value is loaded into a register first.
define i64 @f9(i64 %cmp, i64 *%ptr) {
; CHECK: f9:
; CHECK: lghi [[SWAP:%r[0-9]+]], 1002
; CHECK: csg %r2, [[SWAP]], 0(%r3)
; CHECK: br %r14
  %val = cmpxchg i64 *%ptr, i64 %cmp, i64 1002 seq_cst
  ret i64 %val
}
