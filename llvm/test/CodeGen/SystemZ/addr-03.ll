; Test constant addresses, unlikely as they are.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1() {
; CHECK: f1:
; CHECK: lb %r0, 0
; CHECK: br %r14
  %ptr = inttoptr i64 0 to i8 *
  %val = load volatile i8 *%ptr
  ret void
}

define void @f2() {
; CHECK: f2:
; CHECK: lb %r0, -524288
; CHECK: br %r14
  %ptr = inttoptr i64 -524288 to i8 *
  %val = load volatile i8 *%ptr
  ret void
}

define void @f3() {
; CHECK: f3:
; CHECK-NOT: lb %r0, -524289
; CHECK: br %r14
  %ptr = inttoptr i64 -524289 to i8 *
  %val = load volatile i8 *%ptr
  ret void
}

define void @f4() {
; CHECK: f4:
; CHECK: lb %r0, 524287
; CHECK: br %r14
  %ptr = inttoptr i64 524287 to i8 *
  %val = load volatile i8 *%ptr
  ret void
}

define void @f5() {
; CHECK: f5:
; CHECK-NOT: lb %r0, 524288
; CHECK: br %r14
  %ptr = inttoptr i64 524288 to i8 *
  %val = load volatile i8 *%ptr
  ret void
}
