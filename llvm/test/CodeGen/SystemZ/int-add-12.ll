; Test 64-bit additions of constants to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check additions of 1.
define void @f1(i64 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: agsi 0(%r2), 1
; CHECK: br %r14
  %val = load i64 *%ptr
  %add = add i64 %val, 127
  store i64 %add, i64 *%ptr
  ret void
}

; Check the high end of the constant range.
define void @f2(i64 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: agsi 0(%r2), 127
; CHECK: br %r14
  %val = load i64 *%ptr
  %add = add i64 %val, 127
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next constant up, which must use an addition and a store.
; Both LG/AGHI and LGHI/AG would be OK.
define void @f3(i64 *%ptr) {
; CHECK-LABEL: f3:
; CHECK-NOT: agsi
; CHECK: stg %r0, 0(%r2)
; CHECK: br %r14
  %val = load i64 *%ptr
  %add = add i64 %val, 128
  store i64 %add, i64 *%ptr
  ret void
}

; Check the low end of the constant range.
define void @f4(i64 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: agsi 0(%r2), -128
; CHECK: br %r14
  %val = load i64 *%ptr
  %add = add i64 %val, -128
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next value down, with the same comment as f3.
define void @f5(i64 *%ptr) {
; CHECK-LABEL: f5:
; CHECK-NOT: agsi
; CHECK: stg %r0, 0(%r2)
; CHECK: br %r14
  %val = load i64 *%ptr
  %add = add i64 %val, -129
  store i64 %add, i64 *%ptr
  ret void
}

; Check the high end of the aligned AGSI range.
define void @f6(i64 *%base) {
; CHECK-LABEL: f6:
; CHECK: agsi 524280(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64 *%base, i64 65535
  %val = load i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next doubleword up, which must use separate address logic.
; Other sequences besides this one would be OK.
define void @f7(i64 *%base) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: agsi 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64 *%base, i64 65536
  %val = load i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check the low end of the AGSI range.
define void @f8(i64 *%base) {
; CHECK-LABEL: f8:
; CHECK: agsi -524288(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64 *%base, i64 -65536
  %val = load i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check the next doubleword down, which must use separate address logic.
; Other sequences besides this one would be OK.
define void @f9(i64 *%base) {
; CHECK-LABEL: f9:
; CHECK: agfi %r2, -524296
; CHECK: agsi 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64 *%base, i64 -65537
  %val = load i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}

; Check that AGSI does not allow indices.
define void @f10(i64 %base, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: agr %r2, %r3
; CHECK: agsi 8(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 8
  %ptr = inttoptr i64 %add2 to i64 *
  %val = load i64 *%ptr
  %add = add i64 %val, 1
  store i64 %add, i64 *%ptr
  ret void
}
