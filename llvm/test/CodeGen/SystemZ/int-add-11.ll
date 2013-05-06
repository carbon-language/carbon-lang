; Test 32-bit additions of constants to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check additions of 1.
define void @f1(i32 *%ptr) {
; CHECK: f1:
; CHECK: asi 0(%r2), 1
; CHECK: br %r14
  %val = load i32 *%ptr
  %add = add i32 %val, 127
  store i32 %add, i32 *%ptr
  ret void
}

; Check the high end of the constant range.
define void @f2(i32 *%ptr) {
; CHECK: f2:
; CHECK: asi 0(%r2), 127
; CHECK: br %r14
  %val = load i32 *%ptr
  %add = add i32 %val, 127
  store i32 %add, i32 *%ptr
  ret void
}

; Check the next constant up, which must use an addition and a store.
; Both L/AHI and LHI/A would be OK.
define void @f3(i32 *%ptr) {
; CHECK: f3:
; CHECK-NOT: asi
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32 *%ptr
  %add = add i32 %val, 128
  store i32 %add, i32 *%ptr
  ret void
}

; Check the low end of the constant range.
define void @f4(i32 *%ptr) {
; CHECK: f4:
; CHECK: asi 0(%r2), -128
; CHECK: br %r14
  %val = load i32 *%ptr
  %add = add i32 %val, -128
  store i32 %add, i32 *%ptr
  ret void
}

; Check the next value down, with the same comment as f3.
define void @f5(i32 *%ptr) {
; CHECK: f5:
; CHECK-NOT: asi
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32 *%ptr
  %add = add i32 %val, -129
  store i32 %add, i32 *%ptr
  ret void
}

; Check the high end of the aligned ASI range.
define void @f6(i32 *%base) {
; CHECK: f6:
; CHECK: asi 524284(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131071
  %val = load i32 *%ptr
  %add = add i32 %val, 1
  store i32 %add, i32 *%ptr
  ret void
}

; Check the next word up, which must use separate address logic.
; Other sequences besides this one would be OK.
define void @f7(i32 *%base) {
; CHECK: f7:
; CHECK: agfi %r2, 524288
; CHECK: asi 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131072
  %val = load i32 *%ptr
  %add = add i32 %val, 1
  store i32 %add, i32 *%ptr
  ret void
}

; Check the low end of the ASI range.
define void @f8(i32 *%base) {
; CHECK: f8:
; CHECK: asi -524288(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131072
  %val = load i32 *%ptr
  %add = add i32 %val, 1
  store i32 %add, i32 *%ptr
  ret void
}

; Check the next word down, which must use separate address logic.
; Other sequences besides this one would be OK.
define void @f9(i32 *%base) {
; CHECK: f9:
; CHECK: agfi %r2, -524292
; CHECK: asi 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131073
  %val = load i32 *%ptr
  %add = add i32 %val, 1
  store i32 %add, i32 *%ptr
  ret void
}

; Check that ASI does not allow indices.
define void @f10(i64 %base, i64 %index) {
; CHECK: f10:
; CHECK: agr %r2, %r3
; CHECK: asi 4(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4
  %ptr = inttoptr i64 %add2 to i32 *
  %val = load i32 *%ptr
  %add = add i32 %val, 1
  store i32 %add, i32 *%ptr
  ret void
}
