; Test 32-bit GPR accesses to a PC-relative location.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@gsrc16 = global i16 1
@gsrc32 = global i32 1
@gdst16 = global i16 2
@gdst32 = global i32 2
@gsrc16u = global i16 1, align 1, section "foo"
@gsrc32u = global i32 1, align 2, section "foo"
@gdst16u = global i16 2, align 1, section "foo"
@gdst32u = global i32 2, align 2, section "foo"

; Check sign-extending loads from i16.
define i32 @f1() {
; CHECK-LABEL: f1:
; CHECK: lhrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16 *@gsrc16
  %ext = sext i16 %val to i32
  ret i32 %ext
}

; Check zero-extending loads from i16.
define i32 @f2() {
; CHECK-LABEL: f2:
; CHECK: llhrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16 *@gsrc16
  %ext = zext i16 %val to i32
  ret i32 %ext
}

; Check truncating 16-bit stores.
define void @f3(i32 %val) {
; CHECK-LABEL: f3:
; CHECK: sthrl %r2, gdst16
; CHECK: br %r14
  %half = trunc i32 %val to i16
  store i16 %half, i16 *@gdst16
  ret void
}

; Check plain loads and stores.
define void @f4() {
; CHECK-LABEL: f4:
; CHECK: lrl %r0, gsrc32
; CHECK: strl %r0, gdst32
; CHECK: br %r14
  %val = load i32 *@gsrc32
  store i32 %val, i32 *@gdst32
  ret void
}

; Repeat f1 with an unaligned variable.
define i32 @f5() {
; CHECK-LABEL: f5:
; CHECK: lgrl [[REG:%r[0-5]]], gsrc16u
; CHECK: lh %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i16 *@gsrc16u, align 1
  %ext = sext i16 %val to i32
  ret i32 %ext
}

; Repeat f2 with an unaligned variable.
define i32 @f6() {
; CHECK-LABEL: f6:
; CHECK: lgrl [[REG:%r[0-5]]], gsrc16u
; CHECK: llh %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i16 *@gsrc16u, align 1
  %ext = zext i16 %val to i32
  ret i32 %ext
}

; Repeat f3 with an unaligned variable.
define void @f7(i32 %val) {
; CHECK-LABEL: f7:
; CHECK: lgrl [[REG:%r[0-5]]], gdst16u
; CHECK: sth %r2, 0([[REG]])
; CHECK: br %r14
  %half = trunc i32 %val to i16
  store i16 %half, i16 *@gdst16u, align 1
  ret void
}

; Repeat f4 with unaligned variables.
define void @f8() {
; CHECK-LABEL: f8:
; CHECK: larl [[REG:%r[0-5]]], gsrc32u
; CHECK: l [[VAL:%r[0-5]]], 0([[REG]])
; CHECK: larl [[REG:%r[0-5]]], gdst32u
; CHECK: st [[VAL]], 0([[REG]])
; CHECK: br %r14
  %val = load i32 *@gsrc32u, align 2
  store i32 %val, i32 *@gdst32u, align 2
  ret void
}
