; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that extending loads are assembled properly.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: (func $sext_i8_i32
; CHECK: (setlocal @1 (load_s_i8_i32 @0))
define i32 @sext_i8_i32(i8 *%p) {
  %v = load i8, i8* %p
  %e = sext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: (func $zext_i8_i32
; CHECK: (setlocal @1 (load_u_i8_i32 @0))
define i32 @zext_i8_i32(i8 *%p) {
  %v = load i8, i8* %p
  %e = zext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: (func $sext_i16_i32
; CHECK: (setlocal @1 (load_s_i16_i32 @0))
define i32 @sext_i16_i32(i16 *%p) {
  %v = load i16, i16* %p
  %e = sext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: (func $zext_i16_i32
; CHECK: (setlocal @1 (load_u_i16_i32 @0))
define i32 @zext_i16_i32(i16 *%p) {
  %v = load i16, i16* %p
  %e = zext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: (func $sext_i8_i64
; CHECK: (setlocal @1 (load_s_i8_i64 @0))
define i64 @sext_i8_i64(i8 *%p) {
  %v = load i8, i8* %p
  %e = sext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: (func $zext_i8_i64
; CHECK: (setlocal @1 (load_u_i8_i64 @0))
define i64 @zext_i8_i64(i8 *%p) {
  %v = load i8, i8* %p
  %e = zext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: (func $sext_i16_i64
; CHECK: (setlocal @1 (load_s_i16_i64 @0))
define i64 @sext_i16_i64(i16 *%p) {
  %v = load i16, i16* %p
  %e = sext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: (func $zext_i16_i64
; CHECK: (setlocal @1 (load_u_i16_i64 @0))
define i64 @zext_i16_i64(i16 *%p) {
  %v = load i16, i16* %p
  %e = zext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: (func $sext_i32_i64
; CHECK: (setlocal @1 (load_s_i32_i64 @0))
define i64 @sext_i32_i64(i32 *%p) {
  %v = load i32, i32* %p
  %e = sext i32 %v to i64
  ret i64 %e
}

; CHECK-LABEL: (func $zext_i32_i64
; CHECK: (setlocal @1 (load_u_i32_i64 @0))
define i64 @zext_i32_i64(i32 *%p) {
  %v = load i32, i32* %p
  %e = zext i32 %v to i64
  ret i64 %e
}
