; Test 64-bit GPR stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check STG with no displacement.
define void @f1(i64 *%dst, i64 %val) {
; CHECK-LABEL: f1:
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  store i64 %val, i64 *%dst
  ret void
}

; Check the high end of the aligned STG range.
define void @f2(i64 *%dst, i64 %val) {
; CHECK-LABEL: f2:
; CHECK: stg %r3, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 65535
  store i64 %val, i64 *%ptr
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f3(i64 *%dst, i64 %val) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 65536
  store i64 %val, i64 *%ptr
  ret void
}

; Check the high end of the negative aligned STG range.
define void @f4(i64 *%dst, i64 %val) {
; CHECK-LABEL: f4:
; CHECK: stg %r3, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 -1
  store i64 %val, i64 *%ptr
  ret void
}

; Check the low end of the STG range.
define void @f5(i64 *%dst, i64 %val) {
; CHECK-LABEL: f5:
; CHECK: stg %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 -65536
  store i64 %val, i64 *%ptr
  ret void
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(i64 *%dst, i64 %val) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524296
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 -65537
  store i64 %val, i64 *%ptr
  ret void
}

; Check that STG allows an index.
define void @f7(i64 %dst, i64 %index, i64 %val) {
; CHECK-LABEL: f7:
; CHECK: stg %r4, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %dst, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  store i64 %val, i64 *%ptr
  ret void
}
