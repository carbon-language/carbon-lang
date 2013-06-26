; RUN: llc < %s -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=V7
; RUN: llc < %s -mtriple=armv8-linux-gnueabi | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi | FileCheck %s --check-prefix=Vt8
; This tests that MC/asm header conversion is smooth
;
; V7:      .syntax unified
; V7: .eabi_attribute 6, 10
; V7: .eabi_attribute 20, 1
; V7: .eabi_attribute 21, 1
; V7: .eabi_attribute 23, 3
; V7: .eabi_attribute 24, 1
; V7: .eabi_attribute 25, 1

; V8:      .syntax unified
; V8: .eabi_attribute 6, 14

; Vt8:     .syntax unified
; Vt8: .eabi_attribute 6, 14

define i32 @f(i64 %z) {
	ret i32 0
}
