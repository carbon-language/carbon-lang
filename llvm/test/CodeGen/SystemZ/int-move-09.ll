; Test 64-bit GPR accesses to a PC-relative location.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@gsrc16 = global i16 1
@gsrc32 = global i32 1
@gsrc64 = global i64 1
@gdst16 = global i16 2
@gdst32 = global i32 2
@gdst64 = global i64 2

; Check sign-extending loads from i16.
define i64 @f1() {
; CHECK: f1:
; CHECK: lghrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16 *@gsrc16
  %ext = sext i16 %val to i64
  ret i64 %ext
}

; Check zero-extending loads from i16.
define i64 @f2() {
; CHECK: f2:
; CHECK: llghrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16 *@gsrc16
  %ext = zext i16 %val to i64
  ret i64 %ext
}

; Check sign-extending loads from i32.
define i64 @f3() {
; CHECK: f3:
; CHECK: lgfrl %r2, gsrc32
; CHECK: br %r14
  %val = load i32 *@gsrc32
  %ext = sext i32 %val to i64
  ret i64 %ext
}

; Check zero-extending loads from i32.
define i64 @f4() {
; CHECK: f4:
; CHECK: llgfrl %r2, gsrc32
; CHECK: br %r14
  %val = load i32 *@gsrc32
  %ext = zext i32 %val to i64
  ret i64 %ext
}

; Check truncating 16-bit stores.
define void @f5(i64 %val) {
; CHECK: f5:
; CHECK: sthrl %r2, gdst16
; CHECK: br %r14
  %half = trunc i64 %val to i16
  store i16 %half, i16 *@gdst16
  ret void
}

; Check truncating 32-bit stores.
define void @f6(i64 %val) {
; CHECK: f6:
; CHECK: strl %r2, gdst32
; CHECK: br %r14
  %word = trunc i64 %val to i32
  store i32 %word, i32 *@gdst32
  ret void
}

; Check plain loads and stores.
define void @f7() {
; CHECK: f7:
; CHECK: lgrl %r0, gsrc64
; CHECK: stgrl %r0, gdst64
; CHECK: br %r14
  %val = load i64 *@gsrc64
  store i64 %val, i64 *@gdst64
  ret void
}
