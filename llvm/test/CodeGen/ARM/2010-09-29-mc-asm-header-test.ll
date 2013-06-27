; RUN: llc < %s -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=V7
; RUN: llc < %s -mtriple=armv8-linux-gnueabi | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi | FileCheck %s --check-prefix=Vt8
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=+v8fp | FileCheck %s --check-prefix=V8-V8FP
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=+neon | FileCheck %s --check-prefix=V8-NEON
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

; V8-V8FP:      .syntax unified
; V8-V8FP: .eabi_attribute 6, 14
; V8-V8FP: .eabi_attribute 10, 7

; V8-NEON:      .syntax unified
; V8-NEON: .eabi_attribute 6, 14
; V8-NEON: .eabi_attribute 12, 3

define i32 @f(i64 %z) {
	ret i32 0
}
