; Test 32-bit GPR accesses to a PC-relative location.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@gsrc16 = global i16 1
@gsrc32 = global i32 1
@gdst16 = global i16 2
@gdst32 = global i32 2

; Check sign-extending loads from i16.
define i32 @f1() {
; CHECK: f1:
; CHECK: lhrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16 *@gsrc16
  %ext = sext i16 %val to i32
  ret i32 %ext
}

; Check zero-extending loads from i16.
define i32 @f2() {
; CHECK: f2:
; CHECK: llhrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16 *@gsrc16
  %ext = zext i16 %val to i32
  ret i32 %ext
}

; Check truncating 16-bit stores.
define void @f3(i32 %val) {
; CHECK: f3:
; CHECK: sthrl %r2, gdst16
; CHECK: br %r14
  %half = trunc i32 %val to i16
  store i16 %half, i16 *@gdst16
  ret void
}

; Check plain loads and stores.
define void @f4() {
; CHECK: f4:
; CHECK: lrl %r0, gsrc32
; CHECK: strl %r0, gdst32
; CHECK: br %r14
  %val = load i32 *@gsrc32
  store i32 %val, i32 *@gdst32
  ret void
}
